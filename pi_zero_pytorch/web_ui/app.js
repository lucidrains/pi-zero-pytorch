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
    const calcValueBtn = document.getElementById('calc-value-btn');
    const calcAdvBtn = document.getElementById('calc-advantage-btn');
    const taskList = document.getElementById('task-list');
    const valueChartContainer = document.getElementById('value-chart-container');
    const advantageChartContainer = document.getElementById('advantage-chart-container');
    const chartTooltip = document.getElementById('chart-tooltip');

    const calcStatsBtn = document.getElementById('calc-stats-btn');
    const binarizeBtn = document.getElementById('binarize-btn');
    const statsQuantileInput = document.getElementById('stats-quantile-input');
    const statsResult = document.getElementById('stats-result');

    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingStatus = document.getElementById('loading-status');
    const loadingProgress = document.getElementById('loading-progress');
    const loadingDetail = document.getElementById('loading-detail');

    const gaeGammaInput = document.getElementById('gae-gamma-input');
    const gaeGammaSlider = document.getElementById('gae-gamma-slider');
    const gaeLamInput = document.getElementById('gae-lam-input');
    const gaeLamSlider = document.getElementById('gae-lam-slider');

    let videos = [];
    let labels = {}; // filename -> {task_completed, marked_timestep, returns, value, advantages, advantage_ids}
    let tasks = [];
    let activeVideo = null;
    let currentCutoff = null;

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
                renderCharts(activeVideo.filename);

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

    // Calc Value
    calcValueBtn.onclick = async () => {
        if (!activeVideo) return;
        await calculateValue(activeVideo.filename);
    };

    // Calc Advantage
    calcAdvBtn.onclick = async () => {
        if (!activeVideo) return;
        await calculateAdvantage(activeVideo.filename);
    };

    calcStatsBtn.onclick = async () => {
        const percentile = parseFloat(statsQuantileInput.value);
        calcStatsBtn.disabled = true;
        calcStatsBtn.textContent = 'Calculating...';
        try {
            const response = await fetch('/api/advantage/stats', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ percentile })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                statsResult.textContent = `Cutoff (${data.count} pts): ${data.cutoff.toFixed(6)}`;
                currentCutoff = data.cutoff;
            } else {
                statsResult.textContent = `Error: ${data.error}`;
            }
        } catch (error) {
            console.error('Stats calculation failed:', error);
            statsResult.textContent = 'Error: Check console';
        } finally {
            calcStatsBtn.disabled = false;
            calcStatsBtn.textContent = 'Calc Global Stats';
        }
    };

    binarizeBtn.onclick = async () => {
        if (!activeVideo || currentCutoff === null) return;

        binarizeBtn.disabled = true;
        try {
            const response = await fetch('/api/advantage/binarize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: activeVideo.filename,
                    cutoff: currentCutoff
                })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                if (labels[activeVideo.filename]) {
                    labels[activeVideo.filename].advantage_ids = data.advantage_ids;
                }
                renderTimeline(activeVideo.frames, activeVideo.filename);
            }
        } catch (error) {
            console.error('Binarization failed:', error);
        } finally {
            binarizeBtn.disabled = false;
        }
    };

    // Hyperparameter Sync
    function setupSync(input, slider) {
        input.oninput = () => { slider.value = input.value; };
        slider.oninput = () => { input.value = slider.value; };
    }

    setupSync(gaeGammaInput, gaeGammaSlider);
    setupSync(gaeLamInput, gaeLamSlider);

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
                    labels[filename].value = null;
                    labels[filename].advantages = null;
                }
                if (activeVideo && activeVideo.filename === filename) {
                    renderTimeline(activeVideo.frames, activeVideo.filename);
                    renderTasks();
                    renderCharts(filename);
                }
            }
        } catch (error) {
            console.error('Calculation failed:', error);
        }
    }

    async function calculateValue(filename) {
        calcValueBtn.disabled = true;
        calcValueBtn.textContent = 'Calculating...';
        try {
            const response = await fetch('/api/episode/value/calculate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                if (labels[filename]) {
                    labels[filename].value = data.value;
                    if (activeVideo && activeVideo.filename === filename) {
                        renderCharts(filename);
                    }
                }
            }
        } catch (error) {
            console.error('Value calculation failed:', error);
        } finally {
            calcValueBtn.disabled = false;
            calcValueBtn.textContent = 'Calc Value';
        }
    }

    async function calculateAdvantage(filename) {
        calcAdvBtn.disabled = true;
        calcAdvBtn.textContent = 'Calculating...';
        try {
            const response = await fetch('/api/episode/advantage/calculate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename,
                    gamma: parseFloat(gaeGammaInput.value),
                    lam: parseFloat(gaeLamInput.value)
                })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                if (labels[filename]) {
                    labels[filename].advantages = data.advantages;
                    labels[filename].value = data.value;
                    if (activeVideo && activeVideo.filename === filename) {
                        renderCharts(filename);
                    }
                }
            }
        } catch (error) {
            console.error('Advantage calculation failed:', error);
        } finally {
            calcAdvBtn.disabled = false;
            calcAdvBtn.textContent = 'Calc Advantage';
        }
    }

    async function fetchLabels() {
        const res = await fetch('/api/labels');
        labels = await res.json();
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

        for (let i = 0; i < numFrames; i++) {
            const container = document.createElement('div');
            container.className = 'frame-container';

            const indicator = document.createElement('div');
            indicator.className = 'advantage-indicator';

            if (label && label.advantage_ids && label.advantage_ids[i] !== -1) {
                const advId = label.advantage_ids[i];
                indicator.classList.add(advId === 1 ? 'pos' : 'neg');
                indicator.title = `Advantage ID: ${advId}`;
            }

            const box = document.createElement('div');
            box.className = 'frame-box';

            if (label && label.returns) {
                const ret = label.returns[i];
                if (ret !== null && !isNaN(ret)) {
                    const isSuccess = label.task_completed === 1;
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

            container.appendChild(indicator);
            container.appendChild(box);
            timelineContainer.appendChild(container);
        }
    }

    function renderCharts(filename) {
        const label = labels[filename];
        renderChart(valueChartContainer, label?.value, 'Value', 'value');
        renderChart(advantageChartContainer, label?.advantages, 'Advantage', 'advantage');
    }

    function renderChart(container, data, title, type) {
        if (!data || data.length === 0) {
            container.innerHTML = '';
            container.style.display = 'none';
            return;
        }

        container.style.display = 'block';
        container.innerHTML = `<div class="chart-title">${title}</div>`;

        const validData = data.filter(v => v !== null && !isNaN(v));
        if (validData.length === 0) {
            container.innerHTML = '';
            container.style.display = 'none';
            return;
        }

        const min = Math.min(...validData);
        const max = Math.max(...validData);
        const range = max - min || 0.1;

        const containerWidth = container.clientWidth;
        const width = Math.max(100, containerWidth - 80); // horizontal padding (2rem each side)
        const height = container.clientHeight;

        const points = data.map((v, i) => {
            if (v === null || isNaN(v)) return null;
            const x = (i / (data.length - 1)) * width;
            const y = height - ((v - min) / range) * (height - 20) - 10;
            return { x, y, value: v, index: i };
        });

        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("class", "chart-svg");
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

        // Axis
        const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
        axis.setAttribute("class", "chart-axis");
        axis.setAttribute("x1", "0");
        axis.setAttribute("y1", (height - 1).toString());
        axis.setAttribute("x2", width.toString());
        axis.setAttribute("y2", (height - 1).toString());
        svg.appendChild(axis);

        // Group points into continuous segments (islands)
        const segments = [];
        let currentSegment = [];

        points.forEach(p => {
            if (p) {
                currentSegment.push(p);
            } else {
                if (currentSegment.length > 0) {
                    segments.push(currentSegment);
                    currentSegment = [];
                }
            }
        });
        if (currentSegment.length > 0) segments.push(currentSegment);

        // Render each segment
        segments.forEach(seg => {
            if (seg.length < 1) return;

            const pointsStr = seg.map(p => `${p.x},${p.y}`).join(' ');

            if (seg.length > 1) {
                // Area fill
                const first = seg[0];
                const last = seg[seg.length - 1];
                const areaPoints = `${first.x},${height} ${pointsStr} ${last.x},${height}`;

                const area = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
                area.setAttribute("class", `chart-area-${type}`);
                area.setAttribute("points", areaPoints);
                svg.appendChild(area);

                // Line
                const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
                line.setAttribute("class", `chart-line-${type}`);
                line.setAttribute("points", pointsStr);
                svg.appendChild(line);

                // Dots at each point
                seg.forEach(p => {
                    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                    circle.setAttribute("class", `chart-line-${type}`);
                    circle.setAttribute("cx", p.x.toString());
                    circle.setAttribute("cy", p.y.toString());
                    circle.setAttribute("r", "2.5");
                    circle.setAttribute("fill", type === 'value' ? '#38bdf8' : '#ef4444');
                    circle.setAttribute("stroke", "#fff");
                    circle.setAttribute("stroke-width", "0.5");
                    svg.appendChild(circle);
                });
            } else {
                // Single point case - render a slightly larger circle
                const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                dot.setAttribute("class", `chart-line-${type}`);
                dot.setAttribute("cx", seg[0].x.toString());
                dot.setAttribute("cy", seg[0].y.toString());
                dot.setAttribute("r", "3.0");
                dot.setAttribute("fill", type === 'value' ? '#38bdf8' : '#ef4444');
                dot.setAttribute("stroke", "#fff");
                dot.setAttribute("stroke-width", "1");
                svg.appendChild(dot);
            }
        });

        // Guide line
        const guide = document.createElementNS("http://www.w3.org/2000/svg", "line");
        guide.setAttribute("id", `${type}-guide`);
        guide.setAttribute("class", "chart-guide");
        guide.setAttribute("x1", 0);
        guide.setAttribute("y1", 0);
        guide.setAttribute("x2", 0);
        guide.setAttribute("y2", height);
        svg.appendChild(guide);

        container.appendChild(svg);

        const handleMove = (e) => {
            const rect = svg.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const frameIdx = Math.round((x / width) * (data.length - 1));

            if (frameIdx >= 0 && frameIdx < data.length) {
                const p = points[frameIdx];
                if (p) {
                    showGlobalTooltip(e, frameIdx, p.value, title);
                    syncGuides(frameIdx, width, data.length);
                }
            }
        };

        const handleLeave = () => {
            chartTooltip.style.display = 'none';
            document.querySelectorAll('.chart-guide').forEach(g => g.style.display = 'none');
        };

        svg.addEventListener('mousemove', handleMove);
        svg.addEventListener('mouseleave', handleLeave);
        container.addEventListener('click', (e) => {
            const rect = svg.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const frameIdx = Math.round((x / width) * (data.length - 1));
            if (frameIdx >= 0 && frameIdx < data.length) {
                jumpToFrame(frameIdx);
            }
        });
    }

    function showGlobalTooltip(e, frameIdx, value, title) {
        chartTooltip.style.display = 'block';
        chartTooltip.style.left = `${e.clientX + 15}px`;
        chartTooltip.style.top = `${e.clientY - 40}px`;
        chartTooltip.innerHTML = `<strong>Frame ${frameIdx}</strong><br/>${title}: ${value.toFixed(4)}`;
    }

    function syncGuides(frameIdx, width, totalFrames) {
        const x = (frameIdx / (totalFrames - 1)) * width;
        document.querySelectorAll('.chart-guide').forEach(g => {
            g.setAttribute('x1', x);
            g.setAttribute('x2', x);
            g.style.display = 'block';
        });
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
                    returns: data.returns,
                    value: null,
                    advantages: null
                };
                updateHeaderStatus(filename);
                renderList();
                renderTimeline(activeVideo.frames, filename);
                renderTasks(); // update assign buttons visibility
                renderCharts(filename);

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
        renderCharts(video.filename);
        renderTasks();
        fetchFrames(video.filename);
    }

    async function checkConversionStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();

            if (status.is_converting) {
                loadingOverlay.classList.remove('hidden');
                const percent = status.total > 0 ? (status.progress / status.total) * 100 : 0;
                loadingProgress.style.width = `${percent}%`;
                loadingStatus.textContent = `Converting videos... (${status.progress}/${status.total})`;
                loadingDetail.textContent = status.current_video;

                setTimeout(checkConversionStatus, 1000);
            } else {
                loadingOverlay.classList.add('hidden');
                // Only fetch data once conversion is complete
                await fetchData();
            }
        } catch (error) {
            console.error('Failed to check status:', error);
            setTimeout(checkConversionStatus, 2000);
        }
    }

    checkConversionStatus();
});
