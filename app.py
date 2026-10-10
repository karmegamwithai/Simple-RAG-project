import os

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from werkzeug.utils import secure_filename

from rag import (
    create_vector_store,
    answer_question
)


app = Flask(__name__)


UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {"pdf"}

MAX_FILE_SIZE = 20 * 1024 * 1024


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


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
    return render_template("chat.html")

# =========================================================
# PDF UPLOAD
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    try:

        if "file" not in request.files:

            return jsonify({
                "error": "No file uploaded."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "error": "Please select a PDF."
            }), 400

        if not allowed_file(file.filename):

            return jsonify({
                "error": "Only PDF files are supported."
            }), 400

        filename = secure_filename(
            file.filename or ""
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        chunks = create_vector_store(
            filepath,
            filename
        )

        return jsonify({
            "message": "PDF processed successfully.",
            "filename": filename,
            "chunks": chunks
        })

    except Exception as error:

        print(error)

        return jsonify({
            "error": str(error)
        }), 500


# =========================================================
# ASK QUESTION
# =========================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "Invalid request."
            }), 400

        question = data.get(
            "question",
            ""
        ).strip()

        if not question:

            return jsonify({
                "error": "Question cannot be empty."
            }), 400

        result = answer_question(
            question
        )

        return jsonify(result)

    except Exception as error:

        print(error)

        return jsonify({
            "error": str(error)
        }), 500


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )