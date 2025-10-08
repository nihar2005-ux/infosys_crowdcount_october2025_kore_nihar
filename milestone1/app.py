from flask import Flask, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = "secret123"   # needed for session

# Temporary in-memory "database"
users = {}

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if username in users:
            return "User already exists! <a href='/register'>Try again</a>"
        else:
            users[username] = {"email": email, "password": password}
            return redirect(url_for('login'))
    return render_template('registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users.get(username)
        if user and user['password'] == password:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials! <a href='/login'>Try again</a>"
    return render_template('login.html')


@app.route('/top')
def top():
    user = request.args.get('user', 'Guest')
    return render_template('top.html', user=user)

@app.route('/left')
def left():
    user = request.args.get('user', 'Guest')
    return render_template('left.html', user=user)

# @app.route('/right')
# def right():
#     user = request.args.get('user','user')
#     return render_template('right.html', user=user)
# @app.route('/right')
# def right():
#     if 'user' in session:
#         username = session['user']
#         return render_template('right.html', user=username)
#     return redirect(url_for('login'))
@app.route('/right')
def right():
    if 'user' in session:
        username = session['user']
        return render_template('right.html', user=username)
    return redirect(url_for('login'))



@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html', user=session['user'])
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/profile')
def profile():
    if 'user' not in session:  # check session
        return redirect(url_for('login'))

    username = session['user']
    user_data = users.get(username)
    if not user_data:
        return redirect(url_for('login'))

    return render_template('profile.html', user=username, email=user_data['email'])



if __name__ == "__main__":
    app.run(debug=True)
