document.addEventListener('DOMContentLoaded', () => {
    const videoList = document.getElementById('video-list');
    const player = document.getElementById('main-player');
    const currentFilename = document.getElementById('current-filename');
    const currentFrames = document.getElementById('current-frames');
    const carouselTrack = document.getElementById('carousel-track');

    let videos = [];

    async function fetchVideos() {
        try {
            const response = await fetch('/api/videos');
            videos = await response.json();
            renderList();

            // Auto play first video if available
            if (videos.length > 0) {
                playVideo(videos[0]);
            }
        } catch (error) {
            console.error('Failed to fetch videos:', error);
        }
    }

    async function fetchFrames(filename) {
        carouselTrack.innerHTML = '<div class="loader">Extracting frames...</div>';
        try {
            const response = await fetch(`/api/video/${filename}/frames`);
            const data = await response.json();
            renderCarousel(data.frames);
        } catch (error) {
            console.error('Failed to fetch frames:', error);
            carouselTrack.innerHTML = '<div class="error">Error loading frames</div>';
        }
    }

    function renderList() {
        videoList.innerHTML = '';
        videos.forEach((video, index) => {
            const li = document.createElement('li');
            li.className = 'video-item';
            li.innerHTML = `<span class="item-name">${video.filename}</span>`;
            li.onclick = () => playVideo(video, li);
            videoList.appendChild(li);
        });
    }

    function renderCarousel(frames) {
        carouselTrack.innerHTML = '';
        frames.forEach((frameUrl, index) => {
            const card = document.createElement('div');
            card.className = 'frame-card';

            card.innerHTML = `
                <img src="${frameUrl}" alt="Frame ${index}">
                <div class="labeller-btns">
                    <button class="btn btn-up" title="Good">👍</button>
                    <button class="btn btn-down" title="Bad">👎</button>
                </div>
            `;

            const upBtn = card.querySelector('.btn-up');
            const downBtn = card.querySelector('.btn-down');

            upBtn.onclick = () => {
                upBtn.classList.toggle('active');
                downBtn.classList.remove('active');
            };

            downBtn.onclick = () => {
                downBtn.classList.toggle('active');
                upBtn.classList.remove('active');
            };

            carouselTrack.appendChild(card);
        });
    }

    function playVideo(video, element) {
        // Update active class
        document.querySelectorAll('.video-item').forEach(el => el.classList.remove('active'));
        if (element) {
            element.classList.add('active');
        } else {
            // find by filename if element not provided (initial load)
            const items = document.querySelectorAll('.video-item');
            const idx = videos.findIndex(v => v.filename === video.filename);
            if (idx !== -1 && items[idx]) items[idx].classList.add('active');
        }

        // Update player
        player.src = video.url;
        player.play();

        // Update info
        currentFilename.textContent = video.filename;
        currentFrames.textContent = `${video.frames} frames`;

        // Load frames
        fetchFrames(video.filename);
    }

    fetchVideos();
});
