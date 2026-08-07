import streamlit as st

from document_processor import (
    extract_document_text,
    extract_pdf_pages,
)
from embedding_generator import generate_embeddings
from text_chunker import (
    split_pdf_pages_into_chunks,
    split_text_into_chunks,
)
from vector_store import (
    clear_vector_database,
    create_document_id,
    delete_document,
    get_stored_chunk_count,
    get_stored_documents,
    is_document_stored,
    retrieve_relevant_chunks,
    store_document_chunks,
)
from gemini_service import (
    generate_grounded_answer,
    rewrite_question_with_history,
)


TOP_K_RESULTS = 4

def initialize_chat_history() -> None:
    """Create an empty chat history for a new session."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

@st.cache_data(show_spinner=False)
def prepare_document(
    file_name: str,
    file_bytes: bytes,
):
    """
    Extract text, create chunks, preserve PDF page numbers,
    and generate embeddings.
    """

    extracted_text, metadata = extract_document_text(
        file_name=file_name,
        file_bytes=file_bytes,
    )

    # PDF documents are processed page by page
    # so that each chunk keeps its source page number.
    if metadata["file_type"] == "PDF":
        pages = extract_pdf_pages(file_bytes)

        chunk_records = split_pdf_pages_into_chunks(
            pages
        )

        chunks = [
            record["content"]
            for record in chunk_records
        ]

        page_numbers = [
            record["page_number"]
            for record in chunk_records
        ]

    # DOCX and TXT do not currently use page numbers.
    else:
        chunks = split_text_into_chunks(
            extracted_text
        )

        page_numbers = [
            None
            for _ in chunks
        ]

    embeddings = generate_embeddings(chunks)

    return (
        extracted_text,
        metadata,
        chunks,
        embeddings,
        page_numbers,
    )

def display_sources(
    retrieved_chunks: list[dict],
) -> None:
    """Display the sources supporting an answer."""

    if not retrieved_chunks:
        return

    st.markdown("#### Sources")

    for result in retrieved_chunks:
        source_number = result["rank"]
        source_file = result["source_file"]
        chunk_number = result["chunk_index"] + 1
        page_number = result.get("page_number", 0)

        if page_number:
            source_label = (
                f"Source {source_number}: "
                f"{source_file} — "
                f"Page {page_number} — "
                f"Chunk {chunk_number}"
            )
        else:
            source_label = (
                f"Source {source_number}: "
                f"{source_file} — "
                f"Chunk {chunk_number}"
            )

        with st.expander(
            source_label,
            expanded=False,
        ):
            st.caption(
                f"Vector distance: "
                f"{result['distance']:.4f}"
            )

            st.write(result["content"])

def display_chat_history() -> None:
    """Display all messages saved in the current session."""

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant":
                interpreted_question = message.get(
                    "interpreted_question"
                )

                if interpreted_question:
                    st.caption(
                        "Interpreted question: "
                        f"{interpreted_question}"
                    )

                display_sources(
                    message.get("sources", [])
                )

def display_uploaded_documents(uploaded_files) -> None:
    """
    Automatically process and store uploaded documents.
    """

    st.subheader("Uploaded Documents")

    for file_index, uploaded_file in enumerate(
        uploaded_files
    ):
        try:
            file_bytes = uploaded_file.getvalue()

            document_id = create_document_id(
                file_bytes
            )

            already_stored = is_document_stored(
                document_id
            )

            with st.spinner(
                f"Processing {uploaded_file.name}..."
            ):
                (
                    extracted_text,
                    metadata,
                    chunks,
                    embeddings,
                    page_numbers,
                ) = prepare_document(
                    file_name=uploaded_file.name,
                    file_bytes=file_bytes,
                )

            # Store automatically only when this
            # document is not already indexed.
            if not already_stored:
                with st.spinner(
                    f"Adding {uploaded_file.name} "
                    f"to the knowledge base..."
                ):
                    store_document_chunks(
                        document_id=document_id,
                        file_name=uploaded_file.name,
                        file_type=metadata["file_type"],
                        page_count=metadata["page_count"],
                        chunks=chunks,
                        embeddings=embeddings,
                        page_numbers=page_numbers,
                    )

                storage_status = (
                    "✅ Document processed and ready"
                )

            else:
                storage_status = (
                    "✅ Document already available"
                )

            with st.expander(
                f"📄 {uploaded_file.name}",
                expanded=False,
            ):
                type_column, chunk_column, vector_column = (
                    st.columns(3)
                )

                with type_column:
                    st.metric(
                        label="File type",
                        value=metadata["file_type"],
                    )

                with chunk_column:
                    st.metric(
                        label="Chunks",
                        value=len(chunks),
                    )

                with vector_column:
                    vector_size = (
                        embeddings.shape[1]
                        if embeddings.size
                        else 0
                    )

                    st.metric(
                        label="Vector size",
                        value=vector_size,
                    )

                st.success(storage_status)

                st.write(
                    f"**Extracted characters:** "
                    f"{len(extracted_text):,}"
                )

                if chunks:
                    st.write(
                        "**First chunk preview:**"
                    )

                    st.text_area(
                        label="First chunk",
                        value=chunks[0],
                        height=160,
                        disabled=True,
                        key=(
                            f"chunk_preview_"
                            f"{file_index}"
                        ),
                        label_visibility="collapsed",
                    )

                else:
                    st.warning(
                        "No readable text was found "
                        "in this document."
                    )

        except Exception as error:
            st.error(
                f"Unable to process "
                f"{uploaded_file.name}: {error}"
            )

def process_user_question(
    user_question: str,
) -> None:
    """
    Interpret the question, retrieve relevant chunks,
    generate an answer and save the conversation.
    """

    # Capture the earlier conversation before adding
    # the current question.
    previous_messages = list(
        st.session_state.messages
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        try:
            standalone_question = user_question.strip()

            # Rewriting is required only when previous
            # conversation messages are available.
            if previous_messages:
                try:
                    with st.spinner(
                        "Understanding the follow-up question..."
                    ):
                        standalone_question = (
                            rewrite_question_with_history(
                                question=user_question,
                                chat_history=previous_messages,
                            )
                        )

                except Exception:
                    # Continue using the original question
                    # if rewriting temporarily fails.
                    standalone_question = (
                        user_question.strip()
                    )

            question_was_rewritten = (
                standalone_question.casefold()
                != user_question.strip().casefold()
            )

            interpreted_question = (
                standalone_question
                if question_was_rewritten
                else None
            )

            if interpreted_question:
                st.caption(
                    "Interpreted question: "
                    f"{interpreted_question}"
                )

            with st.spinner(
                "Searching the documents..."
            ):
                retrieved_chunks = retrieve_relevant_chunks(
                    question=standalone_question,
                    top_k=TOP_K_RESULTS,
                )

            if not retrieved_chunks:
                answer = (
                    "No document chunks are stored. "
                    "Please upload and store a document first."
                )

                st.warning(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": [],
                        "interpreted_question": (
                            interpreted_question
                        ),
                    }
                )

                return

            with st.spinner(
                "Generating an answer with Gemini..."
            ):
                answer = generate_grounded_answer(
                    question=standalone_question,
                    retrieved_chunks=retrieved_chunks,
                )

            st.markdown(answer)

            display_sources(retrieved_chunks)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved_chunks,
                    "interpreted_question": (
                        interpreted_question
                    ),
                }
            )

        except Exception as error:
            error_message = (
                "I could not generate the answer because "
                f"an error occurred: {error}"
            )

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                    "interpreted_question": None,
                }
            )

def main() -> None:
    """Display the semantic retrieval interface."""

    st.set_page_config(
        page_title="RAG Document Chatbot",
        page_icon="📄",
        layout="centered",
    )

    initialize_chat_history()

    if "uploader_version" not in st.session_state:
        st.session_state.uploader_version = 0

    st.title("📄 RAG Document Chatbot")

    st.write(
        "Upload documents and retrieve relevant "
        "information using semantic search."
    )

    with st.sidebar:
        st.header("Upload Documents")

        uploaded_files = st.file_uploader(
            label="Choose PDF, DOCX or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=(
                f"document_uploader_"
                f"{st.session_state.uploader_version}"
            ),
        )

        st.divider()

        stored_chunk_count = get_stored_chunk_count()

        st.metric(
            label="Stored document chunks",
            value=stored_chunk_count,
        )

        stored_documents = get_stored_documents()

        if stored_documents:
            st.markdown("### Stored Documents")

            for document in stored_documents:
                st.write(
                    f"📄 **{document['source_file']}**"
                )

                st.caption(
                    f"{document['file_type']} • "
                    f"{document['chunk_count']} chunks"
                )

            st.divider()

            document_options = {
                f"{document['source_file']} ({document['document_id']})":
                    document["document_id"]
                for document in stored_documents
            }

            selected_document_name = st.selectbox(
                "Select document to delete",
                options=list(document_options.keys()),
            )

            if st.button(
                "Delete Selected Document",
                use_container_width=True,
            ):
                selected_document_id = document_options[
                    selected_document_name
                ]

                delete_document(
                    selected_document_id
                )

                st.session_state.messages = []

                st.session_state.uploader_version += 1

                st.rerun()
        else:
            st.info(
                "No documents are currently stored."
            )

        st.caption(
            "The chunks and embeddings are stored "
            "locally in ChromaDB."
        )

        st.divider()

        if st.button(
            "Clear Chat History",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()

        if st.button(
            "Clear Stored Documents",
            use_container_width=True,
        ):
            clear_vector_database()

            st.session_state.messages = []

            st.session_state.uploader_version += 1

            st.rerun()

    if uploaded_files:
        display_uploaded_documents(uploaded_files)
    else:
        st.info(
            "You can upload new documents from the sidebar. "
            "Previously stored documents remain searchable."
        )

    st.divider()

    st.subheader("RAG Document Chatbot")

    st.info(
        "Ask a question about the stored documents. "
        "The application will retrieve relevant information "
        "and generate an answer using Gemini."
    )

    display_chat_history()

    user_question = st.chat_input(
        "Ask a question about the stored documents"
    )

    if user_question:
        process_user_question(user_question)


if __name__ == "__main__":
    main()