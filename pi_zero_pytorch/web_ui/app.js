document.addEventListener('DOMContentLoaded', () => {
    const videoList = document.getElementById('video-list');
    const player = document.getElementById('main-player');
    const currentFilename = document.getElementById('current-filename');
    const currentFrames = document.getElementById('current-frames');
    const carouselTrack = document.getElementById('carousel-track');
    const statusIcon = document.getElementById('label-status-icon');
    const timeLabel = document.getElementById('label-time-container');
    const penaltyInput = document.getElementById('fail-penalty-input');
    const penaltySlider = document.getElementById('fail-penalty-slider');

    let videos = [];
    let labels = {}; // filename -> {task_completed, marked_timestep}
    let activeVideo = null;

    // Penalty sync
    penaltyInput.oninput = () => {
        penaltySlider.value = penaltyInput.value;
    };
    penaltySlider.oninput = () => {
        penaltyInput.value = penaltySlider.value;
    };

    async function fetchData() {
        try {
            const [videoRes, labelRes] = await Promise.all([
                fetch('/api/videos'),
                fetch('/api/labels')
            ]);
            videos = await videoRes.json();
            labels = await labelRes.json();

            renderList();

            if (videos.length > 0) {
                playVideo(videos[0]);
            }
        } catch (error) {
            console.error('Failed to fetch data:', error);
        }
    }

    async function fetchFrames(filename) {
        carouselTrack.innerHTML = '<div class="loader">Extracting frames...</div>';
        try {
            const response = await fetch(`/api/video/${filename}/frames`);
            const data = await response.json();
            renderCarousel(data.frames, filename);
        } catch (error) {
            console.error('Failed to fetch frames:', error);
            carouselTrack.innerHTML = '<div class="error">Error loading frames</div>';
        }
    }

    function renderList() {
        videoList.innerHTML = '';
        videos.forEach((video) => {
            const label = labels[video.filename];
            let statusHtml = '';
            if (label) {
                statusHtml = label.task_completed === 1
                    ? '<span class="video-status-icon status-success">✓</span>'
                    : '<span class="video-status-icon status-fail">✗</span>';
            }

            const li = document.createElement('li');
            li.className = 'video-item';
            li.dataset.filename = video.filename;
            li.innerHTML = `
                <span class="item-name">${video.filename}</span>
                ${statusHtml}
            `;
            li.onclick = () => playVideo(video, li);
            videoList.appendChild(li);
        });

        // Maintain active state in list
        if (activeVideo) {
            const items = document.querySelectorAll('.video-item');
            items.forEach(item => {
                if (item.dataset.filename === activeVideo.filename) {
                    item.classList.add('active');
                }
            });
        }
    }

    function renderCarousel(frames, filename) {
        carouselTrack.innerHTML = '';
        const label = labels[filename];

        frames.forEach((frameUrl, index) => {
            const card = document.createElement('div');
            card.className = 'frame-card';

            const isActiveSuccess = label && label.task_completed === 1 && label.marked_timestep === index;
            const isActiveFail = label && label.task_completed === 0 && label.marked_timestep === index;

            card.innerHTML = `
                <img src="${frameUrl}" alt="Frame ${index}">
                <div class="labeller-btns">
                    <button class="btn btn-up ${isActiveSuccess ? 'active' : ''}" title="Success">✓</button>
                    <button class="btn btn-down ${isActiveFail ? 'active' : ''}" title="Failure">✗</button>
                </div>
            `;

            const upBtn = card.querySelector('.btn-up');
            const downBtn = card.querySelector('.btn-down');

            upBtn.onclick = async () => {
                await labelFrame(filename, index, true);
            };

            downBtn.onclick = async () => {
                await labelFrame(filename, index, false);
            };

            carouselTrack.appendChild(card);
        });
    }

    async function labelFrame(filename, timestep, success) {
        const penalty = parseFloat(penaltyInput.value);
        try {
            const response = await fetch('/api/label', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, timestep, success, penalty })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                labels[filename] = { task_completed: success ? 1 : 0, marked_timestep: timestep };
                updateHeaderStatus(filename);
                renderList();
                // We re-render carousel to update active button
                // but we should ideally just toggle classes for performance
                // For now, let's refresh frames if visible
                if (activeVideo && activeVideo.filename === filename) {
                    const frames = Array.from(carouselTrack.querySelectorAll('.frame-card'));
                    frames.forEach((f, i) => {
                        const up = f.querySelector('.btn-up');
                        const down = f.querySelector('.btn-down');
                        if (i === timestep) {
                            if (success) {
                                up.classList.add('active');
                                down.classList.remove('active');
                            } else {
                                down.classList.add('active');
                                up.classList.remove('active');
                            }
                        } else {
                            up.classList.remove('active');
                            down.classList.remove('active');
                        }
                    });
                }
            }
        } catch (error) {
            console.error('Labelling failed:', error);
        }
    }

    function updateHeaderStatus(filename) {
        const label = labels[filename];
        if (label) {
            if (label.task_completed === 1) {
                statusIcon.innerHTML = '✓';
                statusIcon.className = 'status-success';
            } else {
                statusIcon.innerHTML = '✗';
                statusIcon.className = 'status-fail';
            }
            timeLabel.textContent = `T=${label.marked_timestep}`;
            timeLabel.style.display = 'inline-block';
        } else {
            statusIcon.innerHTML = '';
            timeLabel.style.display = 'none';
        }
    }

    function playVideo(video, element) {
        activeVideo = video;
        document.querySelectorAll('.video-item').forEach(el => el.classList.remove('active'));
        if (element) {
            element.classList.add('active');
        } else {
            const items = document.querySelectorAll('.video-item');
            items.forEach(item => {
                if (item.dataset.filename === video.filename) item.classList.add('active');
            });
        }

        player.src = video.url;
        player.play();

        currentFilename.textContent = video.filename;
        currentFrames.textContent = `${video.frames} frames`;

        updateHeaderStatus(video.filename);
        fetchFrames(video.filename);
    }

    fetchData();
});
