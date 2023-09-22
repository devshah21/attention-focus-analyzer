document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');
    const message = document.getElementById('message');

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const username = loginForm.username.value;
        const password = loginForm.password.value;

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password }),
            });

            const data = await response.json();

            if (response.status === 200) {
                message.textContent = data.message;
                // Redirect or perform other actions upon successful login.
            } else {
                message.textContent = data.message;
            }
        } catch (error) {
            console.error('Login error:', error);
        }
    });
});
