document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('startButton');
    const video = document.getElementById('video');

    startButton.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        } catch (error) {
            console.error('Error accessing camera:', error);
        }
    });
});
