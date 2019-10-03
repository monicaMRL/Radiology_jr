from flask import Flask, render_template

rj_app = Flask(__name__)


@rj_app.route("/")
def home():
    return render_template('rj.html')


@rj_app.route("/results")
def results():
    return "Great this worked!"


if __name__ == "__main__":
    rj_app.run(debug=True)
