document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('startButton');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    let model;

    startButton.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            await loadObjectDetectionModel();
        } catch (error) {
            console.error('Error accessing camera:', error);
        }
    });

    async function loadObjectDetectionModel() {
        model = await cocoSsd.load();
        detectObjects();
    }

    async function detectObjects() {
        if (!model) {
            console.warn('Model not loaded yet. Waiting...');
            setTimeout(detectObjects, 1000); // Wait and try again
            return;
        }

        while (true) {
            const predictions = await model.detect(video);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            cocoSsd.drawPredictions(video, predictions, ctx);
            requestAnimationFrame(detectObjects);
        }
    }
});
