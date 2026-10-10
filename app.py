import os
import uuid
import shutil
import tempfile
import threading

from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session
)

from werkzeug.utils import secure_filename


# =========================================================
# CONFIGURATION
# =========================================================

IS_VERCEL = bool(os.getenv("VERCEL"))

BASE_DIR = Path(
    tempfile.gettempdir()
    if IS_VERCEL
    else Path(__file__).resolve().parent
)

UPLOAD_FOLDER = BASE_DIR / "rag_uploads"

# Set this before importing rag.py because the RAG
# module initializes its storage paths during import.
RAG_DATA_DIR = BASE_DIR / "rag_data"

os.environ["RAG_DATA_DIR"] = str(RAG_DATA_DIR)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

RAG_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# IMPORT RAG BACKEND
# =========================================================

from rag import (
    create_vector_store,
    answer_question,
    clear_vector_store
)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "local-development-only-change-this"
)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

app.config["MAX_CONTENT_LENGTH"] = (
    20 * 1024 * 1024
)

ALLOWED_EXTENSIONS = {"pdf"}

rag_lock = threading.Lock()


# =========================================================
# CHECK FILE TYPE
# =========================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# DELETE DOCUMENT AND VECTOR DATA
# =========================================================

def delete_document():

    with rag_lock:

        clear_vector_store()

        for filepath in UPLOAD_FOLDER.glob("*"):

            if filepath.is_file():

                filepath.unlink(
                    missing_ok=True
                )

    session.pop(
        "document_uploaded",
        None
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# CHAT PAGE
# =========================================================

@app.route("/chat")
def chat_page():

    return render_template(
        "chat.html"
    )


# =========================================================
# RESET DOCUMENT
# =========================================================

@app.route(
    "/reset",
    methods=["POST"]
)
def reset_document():

    try:

        delete_document()

        return jsonify({
            "success": True,
            "message": "Document deleted successfully."
        })

    except Exception as error:

        app.logger.exception(
            "Failed to reset document"
        )

        return jsonify({
            "error": str(error)
        }), 500


# =========================================================
# UPLOAD PDF
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    filepath = None

    try:

        if "file" not in request.files:

            return jsonify({
                "error": "No file uploaded."
            }), 400

        file = request.files["file"]

        if not file.filename:

            return jsonify({
                "error": "Please select a PDF."
            }), 400

        if not allowed_file(file.filename):

            return jsonify({
                "error": "Only PDF files are supported."
            }), 400

        filename = secure_filename(
            file.filename
        )

        unique_filename = (
            f"{uuid.uuid4().hex}_{filename}"
        )

        filepath = UPLOAD_FOLDER / unique_filename

        with rag_lock:

            clear_vector_store()

            file.save(
                str(filepath)
            )

            chunks = create_vector_store(
                str(filepath),
                filename
            )

        session["document_uploaded"] = True

        return jsonify({
            "message": "PDF processed successfully.",
            "filename": filename,
            "chunks": chunks
        })

    except Exception as error:

        app.logger.exception(
            "PDF processing failed"
        )

        return jsonify({
            "error": str(error)
        }), 500

    finally:

        # Delete original PDF immediately after processing.
        # FAISS embeddings remain until reset.
        if filepath is not None:

            filepath.unlink(
                missing_ok=True
            )


# =========================================================
# ASK QUESTION
# =========================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    try:

        if not session.get(
            "document_uploaded"
        ):

            return jsonify({
                "error": "Please upload a PDF first."
            }), 400

        data = request.get_json(
            silent=True
        )

        if not isinstance(data, dict):

            return jsonify({
                "error": "Invalid request."
            }), 400

        question = str(
            data.get(
                "question",
                ""
            )
        ).strip()

        if not question:

            return jsonify({
                "error": "Question cannot be empty."
            }), 400

        with rag_lock:

            result = answer_question(
                question
            )

        return jsonify(
            result
        )

    except Exception as error:

        app.logger.exception(
            "Question answering failed"
        )

        return jsonify({
            "error": str(error)
        }), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "environment": (
            "vercel"
            if IS_VERCEL
            else "local"
        )
    })


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=not IS_VERCEL
    )