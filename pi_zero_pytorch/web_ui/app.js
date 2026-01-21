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
    const timelineContainer = document.getElementById('frame-timeline');
    const resetBtn = document.getElementById('reset-btn');
    const calcReturnsBtn = document.getElementById('calc-returns-btn');
    const taskList = document.getElementById('task-list');

    let videos = [];
    let labels = {}; // filename -> {task_completed, marked_timestep}
    let tasks = [];
    let activeVideo = null;

    // Penalty sync
    penaltyInput.oninput = () => {
        penaltySlider.value = penaltyInput.value;
    };
    penaltySlider.oninput = () => {
        penaltyInput.value = penaltySlider.value;
    };

    // Reset Label
    resetBtn.onclick = async () => {
        if (!activeVideo) return;
        try {
            const response = await fetch('/api/label/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: activeVideo.filename })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                delete labels[activeVideo.filename];
                updateHeaderStatus(activeVideo.filename);
                renderList();
                renderTimeline(activeVideo.frames, activeVideo.filename);

                // Refresh carousel active states
                const frames = Array.from(carouselTrack.querySelectorAll('.frame-card'));
                frames.forEach(f => {
                    f.querySelector('.btn-up').classList.remove('active');
                    f.querySelector('.btn-down').classList.remove('active');
                });
            }
        } catch (error) {
            console.error('Reset failed:', error);
        }
    };

    // Calc Returns
    calcReturnsBtn.onclick = async () => {
        if (!activeVideo) return;
        await calculateReturns(activeVideo.filename);
    };

    async function fetchData() {
        try {
            const [videoRes, labelRes, taskRes] = await Promise.all([
                fetch('/api/videos'),
                fetch('/api/labels'),
                fetch('/api/tasks')
            ]);
            videos = await videoRes.json();
            labels = await labelRes.json();
            tasks = await taskRes.json();

            renderList();
            renderTasks();

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

    function renderTasks() {
        taskList.innerHTML = '';
        const activeLabel = activeVideo ? labels[activeVideo.filename] : null;
        const isAssignable = !!activeVideo;

        tasks.forEach(task => {
            const div = document.createElement('div');
            const isActive = activeLabel && activeLabel.task_id === task.id;
            div.className = `task-item ${isActive ? 'active' : ''} ${isAssignable ? 'assignable' : ''}`;

            div.innerHTML = `
                <div class="task-header">
                    <div class="task-name emphasis">${task.name}</div>
                </div>
                <div class="task-details">
                    <div class="task-field">
                        <div class="task-label">Max Duration</div>
                        <div class="task-value">${task.max_duration} frames</div>
                    </div>
                </div>
                <div class="task-footer">
                    <div class="task-slug">${task.id}</div>
                </div>
                <button class="btn-assign" onclick="event.stopPropagation(); assignTask('${task.id}')">
                    ${isActive ? 'Re-assign' : 'Assign to Episode'}
                </button>
            `;
            taskList.appendChild(div);
        });
    }

    async function assignTask(taskId) {
        if (!activeVideo) return;
        try {
            const response = await fetch('/api/episode/task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: activeVideo.filename, task_id: taskId })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                if (!labels[activeVideo.filename]) {
                    labels[activeVideo.filename] = {
                        task_completed: -1,
                        marked_timestep: -1,
                        returns: []
                    };
                }
                labels[activeVideo.filename].task_id = taskId;

                renderList(); // Refresh sidebar to show slug

                // If it was already labelled, we might want to re-calculate returns since normalization might change
                if (labels[activeVideo.filename].marked_timestep !== -1) {
                    await calculateReturns(activeVideo.filename);
                } else {
                    renderTasks();
                }
            }
        } catch (error) {
            console.error('Task assignment failed:', error);
        }
    }

    // Export assignTask to window so onclick works
    window.assignTask = assignTask;

    async function calculateReturns(filename) {
        try {
            const response = await fetch('/api/returns/calculate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                if (labels[filename]) {
                    labels[filename].returns = data.returns;
                }
                if (activeVideo && activeVideo.filename === filename) {
                    renderTimeline(activeVideo.frames, activeVideo.filename);
                    renderTasks();
                }
            }
        } catch (error) {
            console.error('Calculation failed:', error);
        }
    }

    function renderList() {
        videoList.innerHTML = '';
        videos.forEach((video) => {
            const label = labels[video.filename];
            let statusHtml = '';
            if (label && label.task_completed !== -1) {
                statusHtml = label.task_completed === 1
                    ? '<span class="video-status-icon status-success">✓</span>'
                    : '<span class="video-status-icon status-fail">✗</span>';
            }

            const li = document.createElement('li');
            li.className = 'video-item';
            li.dataset.filename = video.filename;
            li.onclick = () => playVideo(video, li);

            const taskTag = label && label.task_id
                ? `<span class="task-tag" title="${label.task_id}">${label.task_id}</span>`
                : '';

            li.innerHTML = `
                <span class="item-name">${video.filename} ${taskTag}</span>
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

    function renderTimeline(numFrames, filename) {
        timelineContainer.innerHTML = '';
        const label = labels[filename];

        // Find max absolute return for visualization scaling if needed, 
        // but the user said "divide by max episode length" which for GAE-like 
        // (t - marked_timestep) / max_duration results in negative values.

        for (let i = 0; i < numFrames; i++) {
            const box = document.createElement('div');
            box.className = 'frame-box';

            if (label && label.returns) {
                const ret = label.returns[i];
                if (ret !== null && !isNaN(ret)) {
                    const isSuccess = label.task_completed === 1;

                    // The returns are now normalized (e.g., -0.5 meaning half way to goal)
                    // We need to adjust intensity logic. Original was 1 + ret / 100.
                    // If ret is normalized, we might want to use a different scale.
                    // Let's assume ret ranges from -1 to 0.
                    const intensity = Math.max(0.1, 1 + ret);

                    if (isSuccess) {
                        box.style.backgroundColor = `rgba(34, 197, 94, ${intensity})`;
                    } else {
                        box.style.backgroundColor = `rgba(239, 68, 68, ${intensity})`;
                    }

                    box.title = `Return: ${ret.toFixed(4)}`;

                    if (ret === 0) {
                        box.classList.add(isSuccess ? 'success' : 'fail');
                    }
                }
            } else if (label && label.marked_timestep === i) {
                box.classList.add(label.task_completed === 1 ? 'success' : 'fail');
            }

            box.onclick = () => {
                jumpToFrame(i);
            };
            timelineContainer.appendChild(box);
        }
    }

    function jumpToFrame(index) {
        if (!activeVideo || !player.duration) return;

        // Accurate seek using frames / duration ratio
        const seekTime = (index / activeVideo.frames) * player.duration;
        player.currentTime = seekTime;

        // Autoscroll carousel
        const cards = carouselTrack.querySelectorAll('.frame-card');
        if (cards[index]) {
            cards[index].scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
            // Highlight current frame in timeline
            document.querySelectorAll('.frame-box').forEach((box, i) => {
                box.classList.toggle('current', i === index);
            });
        }
    }

    function renderCarousel(frames, filename) {
        carouselTrack.innerHTML = '';
        const label = labels[filename];

        frames.forEach((frameUrl, index) => {
            const card = document.createElement('div');
            card.className = 'frame-card';
            card.dataset.index = index;

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

            upBtn.onclick = async (e) => {
                e.stopPropagation();
                await labelFrame(filename, index, true);
            };

            downBtn.onclick = async (e) => {
                e.stopPropagation();
                await labelFrame(filename, index, false);
            };

            card.onclick = () => jumpToFrame(index);

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
                labels[filename] = {
                    task_completed: success ? 1 : 0,
                    marked_timestep: timestep,
                    task_id: labels[filename]?.task_id, // preserve task_id
                    returns: data.returns
                };
                updateHeaderStatus(filename);
                renderList();
                renderTimeline(activeVideo.frames, filename);
                renderTasks(); // update assign buttons visibility

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
        if (label && label.task_completed !== -1) {
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

        const taskBadge = document.getElementById('current-task');
        if (label && label.task_id) {
            taskBadge.textContent = label.task_id;
            taskBadge.style.display = 'inline-block';
        } else {
            taskBadge.style.display = 'none';
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
        player.load();
        player.play();

        currentFilename.textContent = video.filename;
        currentFrames.textContent = `${video.frames} frames`;

        updateHeaderStatus(video.filename);
        renderTimeline(video.frames, video.filename);
        renderTasks();
        fetchFrames(video.filename);
    }

    fetchData();
});
