const express = require('express');
const bodyParser = require('body-parser');
const session = require('express-session');

const crypto = require('crypto');

// Generate a random 32-byte (256-bit) secret key
const secretKey = crypto.randomBytes(32).toString('hex');

const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(session({ secret: secretKey, resave: false, saveUninitialized: true }));

// Hard-coded user data (in practice, use a database)
const users = [
    { id: 1, username: 'user1', password: 'password1' },
    { id: 2, username: 'user2', password: 'password2' },
];

// Middleware to check if the user is authenticated
function isAuthenticated(req, res, next) {
    if (req.session && req.session.userId) {
        return next();
    } else {
        res.status(401).json({ message: 'Unauthorized' });
    }
}

// User login endpoint
app.post('/login', (req, res) => {
    const { username, password } = req.body;

    const user = users.find(u => u.username === username && u.password === password);

    if (user) {
        req.session.userId = user.id;
        res.status(200).json({ message: 'Login successful' });
    } else {
        res.status(401).json({ message: 'Login failed' });
    }
});

// Protected endpoint (requires authentication)
app.get('/protected', isAuthenticated, (req, res) => {
    res.status(200).json({ message: 'Protected resource' });
});

// Logout endpoint
app.post('/logout', (req, res) => {
    if (req.session) {
        req.session.destroy(err => {
            if (err) {
                console.error('Session destroy error:', err);
                res.status(500).json({ message: 'Internal server error' });
            } else {
                res.status(200).json({ message: 'Logout successful' });
            }
        });
    }
});

app.get('/', (req, res) => {
    res.send('Hello, World!');
});



app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
