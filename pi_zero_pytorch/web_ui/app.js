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

    const exportBtn = document.getElementById('export-btn');
    const invalidateBtn = document.getElementById('invalidate-btn');
    const invalidateCutoffInput = document.getElementById('invalidate-cutoff-input');
    const invalidateCutoffSlider = document.getElementById('invalidate-cutoff-slider');

    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingStatus = document.getElementById('loading-status');
    const loadingProgress = document.getElementById('loading-progress');
    const loadingDetail = document.getElementById('loading-detail');

    // Force hide overlay immediately on script load to be safe
    if (loadingOverlay) loadingOverlay.classList.add('hidden-element');

    const gaeGammaInput = document.getElementById('gae-gamma-input');
    const gaeGammaSlider = document.getElementById('gae-gamma-slider');
    const gaeLamInput = document.getElementById('gae-lam-input');
    const gaeLamSlider = document.getElementById('gae-lam-slider');

    let videos = [];
    let labels = {}; // filename -> {task_completed, marked_timestep, returns, value, advantages, advantage_ids, invalidated}
    let tasks = [];
    let activeVideo = null;
    let activePlayers = []; // Store multiple players if num_views > 1
    let currentCutoff = null;
    let recapState = null;
    let viewpointNames = {};
    let activeFolders = []; // Track active folder paths for display

    // Sidebar elements for visibility control
    const rolloutsSidebar = document.querySelector('.sidebar');
    const tasksSidebar = document.querySelector('.task-sidebar');
    const activeFoldersContainer = document.getElementById('active-folders');

    // Update active folders display under Rollouts title
    function updateActiveFoldersDisplay() {
        if (!activeFoldersContainer) return;
        activeFoldersContainer.innerHTML = '';

        for (const folder of activeFolders) {
            const item = document.createElement('div');
            item.className = 'active-folder-item' + (folder.active ? ' active' : '');
            item.innerHTML = `<span class="folder-icon">📁</span><span class="folder-path">${folder.path}</span>`;
            item.onclick = () => {
                // Set this folder as active and reload its data
                activeFolders.forEach(f => f.active = false);
                folder.active = true;
                updateActiveFoldersDisplay();
                updateFolderHighlighting();
            };
            activeFoldersContainer.appendChild(item);
        }
    }

    // Update folder icon highlighting in RECAP task list
    function updateFolderHighlighting() {
        const allFolderItems = document.querySelectorAll('.recap-data-item');
        allFolderItems.forEach(item => {
            item.classList.remove('active');
            // Check if this folder matches any active folder
            const folderText = item.textContent;
            for (const folder of activeFolders) {
                if (folder.active && folderText.includes(folder.id)) {
                    item.classList.add('active');
                }
            }
        });
    }

    // Initial UI state
    function updateUIVisibility() {
        const mainContentElements = [
            timelineContainer,
            valueChartContainer,
            advantageChartContainer,
            document.querySelector('.player-container'),
            document.querySelector('.carousel-container')
        ];

        const isVisible = activeVideo !== null;
        mainContentElements.forEach(el => {
            if (el) {
                if (isVisible) {
                    el.classList.remove('hidden-element');
                } else {
                    el.classList.add('hidden-element');
                }
            }
        });

        // Also header info
        const headerInfoElements = [
            resetBtn,
            document.getElementById('current-frames'),
            document.getElementById('current-task'),
            timeLabel,
            document.querySelector('.action-group')
        ];
        headerInfoElements.forEach(el => {
            if (el) {
                if (isVisible) {
                    el.classList.remove('hidden-element');
                } else {
                    el.classList.add('hidden-element');
                }
            }
        });

        // Persistent actions (like training) should be visible if we have data, even if no active video
        const persistentGroup = document.querySelector('.action-group-persistent');
        if (persistentGroup) {
            // Check if ReplayBuffer exists or if we've loaded a data folder
            // Watch for recapState - if enabled, show persistent actions
            const hasDataLoaded = videos.length > 0 || (recapState && recapState.enabled);
            if (hasDataLoaded) {
                persistentGroup.classList.remove('hidden-element');
            } else {
                persistentGroup.classList.add('hidden-element');
            }
        }

        if (rolloutsSidebar && tasksSidebar) {
            const hasVideos = videos.length > 0;
            const isRecap = recapState && recapState.enabled;

            // Rollouts sidebar (video list): visible if we have videos or if recap is enabled (where it shows folders)
            // Actually, in standalone mode, it's our main navigation.
            rolloutsSidebar.style.display = (hasVideos || isRecap) ? 'block' : 'none';

            // Tasks sidebar (RECAP ops): ONLY in recap mode and if we have data
            tasksSidebar.style.display = (isRecap && hasVideos) ? 'block' : 'none';

            if (!hasVideos) {
                // Show "no trajectories loaded" message
                const mainContent = document.querySelector('.main-content');
                let noTrajMsg = document.getElementById('no-trajectories-msg');
                if (!noTrajMsg) {
                    noTrajMsg = document.createElement('div');
                    noTrajMsg.id = 'no-trajectories-msg';
                    noTrajMsg.className = 'info-message';
                    const message = isRecap ?
                        'Please click on a folder in the RECAP sidebar to start labelling.' :
                        'Please load a folder or select it via command line.';
                    noTrajMsg.innerHTML = `
                            <div class="info-icon">📂</div>
                            <h2>No trajectories loaded</h2>
                            <p>${message}</p>
                        `;
                    mainContent.appendChild(noTrajMsg);
                }
                noTrajMsg.style.display = 'flex';
            } else {
                const noTrajMsg = document.getElementById('no-trajectories-msg');
                if (noTrajMsg) noTrajMsg.style.display = 'none';
            }
        }
    }

    updateUIVisibility();

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
                    f.querySelector('.btn-star').classList.remove('active');
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

    exportBtn.onclick = () => {
        window.location.href = '/api/export';
    };

    calcStatsBtn.onclick = async () => {
        statsResult.classList.remove('hidden-element');
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
        statsResult.classList.remove('hidden-element');
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
                console.log(`[RECAP] Binarization successful for ${activeVideo.filename}. IDs:`, data.advantage_ids);
                if (labels[activeVideo.filename]) {
                    labels[activeVideo.filename].advantage_ids = data.advantage_ids;
                }
                renderTimeline(activeVideo.frames, activeVideo.filename);
                renderRecapSidebar();
                statsResult.textContent = `Binarized with cutoff: ${currentCutoff.toFixed(6)}`;
            }
        } catch (error) {
            console.error('Binarization failed:', error);
            statsResult.textContent = 'Error: Binarization failed';
        } finally {
            binarizeBtn.disabled = false;
        }
    };

    invalidateBtn.onclick = async () => {
        if (!activeVideo) return;
        const cutoff = parseFloat(invalidateCutoffInput.value);
        invalidateBtn.disabled = true;
        invalidateBtn.textContent = 'Invalidating...';

        try {
            const response = await fetch('/api/episode/invalidate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: activeVideo.filename,
                    cutoff: cutoff
                })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                if (labels[activeVideo.filename]) {
                    labels[activeVideo.filename].invalidated = data.invalidated;
                }
                renderTimeline(activeVideo.frames, activeVideo.filename);
            }
        } catch (error) {
            console.error('Invalidation failed:', error);
        } finally {
            invalidateBtn.disabled = false;
            invalidateBtn.textContent = 'Invalidate';
        }
    };

    // Hyperparameter Sync
    function setupSync(input, slider) {
        input.oninput = () => { slider.value = input.value; };
        slider.oninput = () => { input.value = slider.value; };
    }

    setupSync(gaeGammaInput, gaeGammaSlider);
    setupSync(gaeLamInput, gaeLamSlider);
    setupSync(invalidateCutoffInput, invalidateCutoffSlider);

    async function fetchData() {
        console.log("Fetching Rollout data...");
        try {
            const [videoRes, labelRes, taskRes, viewpointRes] = await Promise.all([
                fetch('/api/videos'),
                fetch('/api/labels'),
                fetch('/api/tasks'),
                fetch('/api/viewpoints')
            ]);
            videos = await videoRes.json();
            labels = await labelRes.json();
            tasks = await taskRes.json();
            const vpData = await viewpointRes.json();
            viewpointNames = vpData.viewpoints || {};

            console.log(`Fetch complete: ${videos.length} videos, ${Object.keys(labels).length} labels`);

            renderList();
            renderTasks();
            updateUIVisibility();

            if (videos.length > 0 && !activeVideo) {
                console.log("Auto-playing first video");
                playVideo(videos[0]);
            }
        } catch (error) {
            console.error('Failed to fetch data:', error);
        }

        // If we still have no videos, retry in a bit (to handle initialization race conditions)
        if (videos.length === 0) {
            console.log("No videos found yet, retrying in 2s...");
            setTimeout(fetchData, 2000);
        }
    }
    window.fetchData = fetchData;

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
                if (!labels[filename]) {
                    labels[filename] = { value: [], advantages: [], returns: [], task_completed: -1 };
                }
                labels[filename].value = data.value;
                if (activeVideo && activeVideo.filename === filename) {
                    renderCharts(filename);
                    calcAdvBtn.disabled = false; // Enable advantage calc after value succeeds
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
                if (!labels[filename]) {
                    labels[filename] = { value: [], advantages: [], returns: [], task_completed: -1 };
                }
                labels[filename].advantages = data.advantages;
                labels[filename].value = data.value; // Keep existing line
                labels[filename].advantage_ids = data.advantage_ids;
                if (activeVideo && activeVideo.filename === filename) {
                    renderCharts(filename);
                    renderTimeline(activeVideo.frames, activeVideo.filename); // Changed to use activeVideo.frames
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

            const folderTag = video.folder
                ? `<span class="folder-label">${video.folder}</span>`
                : '';

            li.innerHTML = `
                <div class="video-info-container">
                    <span class="item-name">${video.filename} ${taskTag}</span>
                    ${folderTag}
                </div>
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

            const invalidIndicator = document.createElement('div');
            invalidIndicator.className = 'invalidated-indicator';
            invalidIndicator.style.visibility = (label && label.invalidated) ? 'visible' : 'hidden';

            const expertIndicator = document.createElement('div');
            expertIndicator.className = 'expert-indicator';
            expertIndicator.style.visibility = (label && label.expert_segment && label.expert_segment[i]) ? 'visible' : 'hidden';

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
            container.appendChild(invalidIndicator);
            container.appendChild(box);
            container.appendChild(expertIndicator);

            if (label && label.invalidated) {
                box.classList.add('invalidated');
            }

            timelineContainer.appendChild(container);
        }
    }

    function renderCharts(filename) {
        if (!activeVideo || activeVideo.filename !== filename) return;
        const info = labels[filename];
        if (!info) return;

        // Use activeVideo.frames for total length to ensure correct alignment
        renderChart(valueChartContainer, info.value, 'Value', 'value', activeVideo.frames);
        renderChart(advantageChartContainer, info.advantages, 'Advantage', 'advantage', activeVideo.frames);
    }

    function renderChart(container, data, title, type, totalLength) {
        if (!data || data.length === 0) {
            container.innerHTML = '';
            container.classList.remove('active');
            container.style.display = 'none';
            return;
        }

        container.classList.add('active');
        container.style.display = 'block';
        container.innerHTML = `<div class="chart-title">${title}</div>`;

        const validData = data.filter(v => v !== null && !isNaN(v));
        if (validData.length === 0) {
            container.innerHTML = '';
            container.classList.remove('active');
            container.style.display = 'none';
            return;
        }

        const min = Math.min(...validData);
        const max = Math.max(...validData);
        const range = max - min || 0.1;

        const containerWidth = container.clientWidth || 800;
        const width = Math.max(100, containerWidth - 80);
        const height = container.clientHeight || 100;

        // Alignment fix: use totalLength to map index to X coordinate correctly
        const denom = (totalLength && totalLength > 1) ? (totalLength - 1) : (data.length - 1 || 1);

        const points = data.map((v, i) => {
            if (v === null || isNaN(v)) return null;
            const x = (i / denom) * width;
            const y = height - ((v - min) / range) * (height - 20) - 10;

            if (isNaN(x) || isNaN(y)) return null;
            return { x, y, value: v, index: i };
        });

        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("class", "chart-svg");
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.setAttribute("preserveAspectRatio", "none");

        // Axis
        const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
        axis.setAttribute("class", "chart-axis");
        axis.setAttribute("x1", "0");
        axis.setAttribute("y1", (height - 1).toString());
        axis.setAttribute("x2", width.toString());
        axis.setAttribute("y2", (height - 1).toString());
        svg.appendChild(axis);

        // Zero-line for advantages
        if (type === 'advantage') {
            const zeroValueY = height - ((0 - min) / range) * (height - 20) - 10;
            if (!isNaN(zeroValueY) && zeroValueY >= 0 && zeroValueY <= height) {
                const zeroLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
                zeroLine.setAttribute("x1", "0");
                zeroLine.setAttribute("y1", zeroValueY.toString());
                zeroLine.setAttribute("x2", width.toString());
                zeroLine.setAttribute("y2", zeroValueY.toString());
                zeroLine.setAttribute("stroke", "rgba(255, 255, 255, 0.15)");
                zeroLine.setAttribute("stroke-dasharray", "4 4");
                svg.appendChild(zeroLine);
            }
        }

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
        if (!activeVideo) return;

        const mainPlayer = activePlayers[0] || player;
        if (!mainPlayer || !mainPlayer.duration) return;

        const seekTime = (index / activeVideo.frames) * mainPlayer.duration;

        activePlayers.forEach(p => {
            p.currentTime = seekTime;
        });

        // ensure main player for backward compatibility
        if (activePlayers.length === 0) player.currentTime = seekTime;

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
    window.jumpToFrame = jumpToFrame;

    function renderCarousel(frames, filename) {
        carouselTrack.innerHTML = '';
        const label = labels[filename];

        frames.forEach((frameUrl, index) => {
            const card = document.createElement('div');
            card.className = 'frame-card';
            card.dataset.index = index;

            const isActiveSuccess = label && label.task_completed === 1 && label.marked_timestep === index;
            const isActiveFail = label && label.task_completed === 0 && label.marked_timestep === index;
            const isActiveIntervention = label && label.expert_segment && label.expert_segment[index];

            let framesHtml = '';
            if (Array.isArray(frameUrl)) {
                // Multi-view
                framesHtml = `<div class="frame-stack">`;
                frameUrl.forEach((vUrl, vIdx) => {
                    const vpName = viewpointNames[vIdx] || `View ${vIdx}`;
                    framesHtml += `
                        <div class="view-container">
                            <img src="${vUrl}" alt="View ${vIdx}">
                            <div class="view-label">${vpName}</div>
                        </div>
                    `;
                });
                framesHtml += `</div>`;
            } else {
                // Single-view
                framesHtml = `<img src="${frameUrl}" alt="Frame ${index}">`;
            }

            card.innerHTML = `
                ${framesHtml}
                <div class="labeller-btns">
                    <button class="btn btn-up ${isActiveSuccess ? 'active' : ''}" title="Success">✓</button>
                    <button class="btn btn-down ${isActiveFail ? 'active' : ''}" title="Failure">✗</button>
                    <button class="btn btn-star ${isActiveIntervention ? 'active' : ''}" title="Expert Intervention">★</button>
                </div>
            `;

            const upBtn = card.querySelector('.btn-up');
            const downBtn = card.querySelector('.btn-down');
            const starBtn = card.querySelector('.btn-star');

            upBtn.onclick = async (e) => {
                e.stopPropagation();
                await labelFrame(filename, index, true);
            };

            downBtn.onclick = async (e) => {
                e.stopPropagation();
                await labelFrame(filename, index, false);
            };

            starBtn.onclick = async (e) => {
                e.stopPropagation();
                await labelIntervention(filename, index);
            };

            card.onclick = () => jumpToFrame(index);

            carouselTrack.appendChild(card);
        });
    }

    async function labelIntervention(filename, timestep) {
        try {
            const response = await fetch('/api/label/intervention', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, timestep })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                if (!labels[filename]) labels[filename] = {};
                labels[filename].is_expert_intervention = true;
                labels[filename].expert_segment = data.expert_segment;
                labels[filename].advantage_ids = data.advantage_ids;

                renderList();
                renderTimeline(activeVideo.frames, filename);
                renderCharts(filename);
                updateHeaderStatus(filename);

                // Update carousel buttons
                const frames = Array.from(carouselTrack.querySelectorAll('.frame-card'));
                frames.forEach((f, i) => {
                    const star = f.querySelector('.btn-star');
                    if (data.expert_segment[i]) {
                        star.classList.add('active');
                    } else {
                        star.classList.remove('active');
                    }
                });
            }
        } catch (error) {
            console.error('Intervention labelling failed:', error);
        }
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

        const interventionBadge = document.getElementById('intervention-badge');
        if (label && label.is_expert_intervention) {
            interventionBadge.style.display = 'inline-block';
        } else {
            interventionBadge.style.display = 'none';
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

        const videoWrapper = document.querySelector('.video-wrapper');
        videoWrapper.innerHTML = '';
        activePlayers = [];

        if (video.num_views > 1) {
            const grid = document.createElement('div');
            grid.className = `video-grid views-${video.num_views}`;

            for (let v = 0; v < video.num_views; v++) {
                const item = document.createElement('div');
                item.className = 'player-item';

                const vPlayer = document.createElement('video');
                vPlayer.src = `${video.url}.${v}.mp4`;
                vPlayer.controls = true; // Enable controls for all views
                vPlayer.muted = true;
                vPlayer.playsInline = true;

                const vpName = viewpointNames[v] || `View ${v}`;
                const label = document.createElement('div');
                label.className = 'player-label';
                label.textContent = vpName;

                item.appendChild(vPlayer);
                item.appendChild(label);
                grid.appendChild(item);
                activePlayers.push(vPlayer);
            }
            videoWrapper.appendChild(grid);

            // Sync logic
            const sync = (event) => {
                const target = event.target;
                if (target.seeking_sync) return; // Prevent loop

                activePlayers.forEach(p => {
                    if (p === target) return;

                    if (event.type === 'play') p.play().catch(() => { });
                    if (event.type === 'pause') p.pause();
                    if (event.type === 'ratechange') p.playbackRate = target.playbackRate;

                    if (event.type === 'timeupdate' || event.type === 'seeking') {
                        if (Math.abs(p.currentTime - target.currentTime) > 0.15) {
                            p.seeking_sync = true;
                            p.currentTime = target.currentTime;
                            // Clear flag after a short delay to allow seek to complete
                            setTimeout(() => { p.seeking_sync = false; }, 50);
                        }
                    }
                });
            };

            activePlayers.forEach(p => {
                p.addEventListener('play', sync);
                p.addEventListener('pause', sync);
                p.addEventListener('seeking', sync);
                p.addEventListener('timeupdate', sync);
                p.addEventListener('ratechange', sync);
                p.load();
            });

            activePlayers[0].play().catch(() => { });
        } else {
            const vPlayer = document.createElement('video');
            vPlayer.id = 'main-player';
            vPlayer.src = video.url;
            vPlayer.controls = true;
            vPlayer.muted = true; // Consistency
            videoWrapper.appendChild(vPlayer);
            activePlayers = [vPlayer];
            vPlayer.load();
            vPlayer.play().catch(() => { });
        }

        currentFilename.textContent = video.filename;
        currentFrames.textContent = `${video.frames} frames`;

        updateHeaderStatus(video.filename);
        updateUIVisibility();
        renderTimeline(video.frames, video.filename);
        renderCharts(video.filename);
        renderTasks();
        fetchFrames(video.filename);
    }


    async function fetchStatus() {
        const response = await fetch('/api/status');
        return await response.json();
    }

    let conversionPoller = null;
    async function checkConversionStatus() {
        // Reset flag
        window.__RECAP_LOADING_FINISHED__ = false;

        if (conversionPoller) clearInterval(conversionPoller);

        conversionPoller = setInterval(async () => {
            // If in RECAP mode and no video dir is set, skip conversion check
            if (recapState && recapState.enabled && loadingOverlay.dataset.recapLoading !== 'true') {
                loadingOverlay.classList.add('hidden-element');
                clearInterval(conversionPoller);
                return;
            }

            try {
                const status = await fetchStatus();
                console.log('[POLLER] Status:', status);
                if (status.is_converting) {
                    loadingOverlay.classList.remove('hidden-element');
                    const percent = status.total > 0 ? (status.progress / status.total) * 100 : 0;
                    loadingProgress.style.width = `${percent}%`;
                    loadingStatus.textContent = `Converting videos... (${status.progress}/${status.total})`;
                    loadingDetail.textContent = status.current_video || 'Preparing ReplayBuffer...';
                } else {
                    console.log('[POLLER] Conversion complete, hiding overlay');
                    loadingOverlay.classList.add('hidden-element');
                    loadingOverlay.dataset.recapLoading = 'false';
                    window.__RECAP_LOADING_FINISHED__ = true;
                    clearInterval(conversionPoller);
                    await fetchData();
                }
            } catch (error) {
                console.error('[POLLER] Failed to check status:', error);
            }
        }, 1000);
    }
    window.__RECAP_LOADING_FINISHED__ = false;

    const recapSidebar = document.getElementById('recap-sidebar');
    const recapTaskList = document.getElementById('recap-task-list');
    const recapGeneralistStatus = document.getElementById('recap-generalist-status');
    const recapPolicyStatus = document.getElementById('recap-policy-status');
    const btnPretrain = document.getElementById('btn-pretrain');
    const btnFinetunePolicy = document.getElementById('btn-finetune-policy');



    async function fetchRecapState() {
        try {
            const response = await fetch('/api/recap/state');
            recapState = await response.json();

            if (recapState.enabled) {
                recapSidebar.style.display = 'block';
                renderRecapSidebar();
            } else {
                recapSidebar.style.display = 'none';
            }
            updateUIVisibility();
        } catch (error) {
            console.error('Failed to fetch RECAP state:', error);
        }
    }

    function renderRecapSidebar() {
        if (!recapState || !recapState.enabled) return;

        // Update generalist status
        const isReady = recapState.pretrained.actor && recapState.pretrained.critic;
        recapGeneralistStatus.textContent = isReady ? 'Ready' : 'Not Ready';
        recapGeneralistStatus.className = `recap-status-value ${isReady ? 'success' : 'fail'}`;
        btnPretrain.style.display = isReady ? 'none' : 'block';

        // Update policy status - check if iteration 0 has weights and if we have binarized data
        // For simplicity, we'll just check if actor.pt exists in iterate 0 or 1 etc.
        // But the user specifically wants a button that triggers finetuning.
        // We'll enable the button if we have loaded data and have binarized advantages.
        const activeLabel = activeVideo ? labels[activeVideo.filename] : null;
        const hasBinarized = activeLabel && activeLabel.advantage_ids && activeLabel.advantage_ids.some(id => id !== -1);

        console.log(`[RECAP] renderRecapSidebar: hasBinarized=${hasBinarized}, activeVideo=${activeVideo?.filename}`);

        btnFinetunePolicy.disabled = !hasBinarized;

        // Check if any iteration > 0 has weights or if iter 0 has been "finetuned"
        // Actually, let's just use a simple flag or check the workspace
        const isFinetuned = recapState.tasks.some(t => t.iterations.some(i => i.id > 0 && i.actor));
        recapPolicyStatus.textContent = isFinetuned ? 'Ready' : 'Not Ready';
        recapPolicyStatus.className = `recap-status-value ${isFinetuned ? 'success' : 'fail'}`;
        recapTaskList.innerHTML = '';

        // Render Pretrained Dataset if available
        if (recapState.pretrained_data) {
            const ptGroup = document.createElement('div');
            ptGroup.className = 'recap-task-group';
            ptGroup.innerHTML = `
                <div class="recap-task-header">
                    <span class="recap-task-name">PRETRAINED DATASET</span>
                </div>
                <div class="recap-data-list">
                    <div class="recap-data-item" onclick="loadPretrained()">
                        📁 Original Dataset (${recapState.pretrained_data.video_count} videos)
                    </div>
                </div>
            `;
            recapTaskList.appendChild(ptGroup);
        }

        // Render task list
        for (const task of recapState.tasks) {
            const taskGroup = document.createElement('div');
            taskGroup.className = 'recap-task-group';

            const taskHeader = document.createElement('div');
            taskHeader.className = 'recap-task-header';
            taskHeader.innerHTML = `
                <span class="recap-task-name">${task.name.replace(/_/g, ' ').toUpperCase()}</span>
                ${isReady && task.iterations.length === 0 ?
                    `<button class="btn-recap-mini" onclick="recapSpecialize('${task.name}')">SFT</button>` : ''}
            `;
            taskGroup.appendChild(taskHeader);

            // Render iterations
            for (const iter of task.iterations) {
                const iterRow = document.createElement('div');
                iterRow.className = 'recap-iteration';

                const isTrained = iter.actor && iter.critic;
                const hasData = iter.data.length > 0;
                const indicatorClass = isTrained ? 'indicator-trained' : (hasData ? 'indicator-data' : 'indicator-none');
                const statusText = isTrained ? 'Trained' : (hasData ? 'Collecting' : 'Init');

                iterRow.innerHTML = `
                    <div class="iteration-num">${iter.id}</div>
                    <div class="iteration-status">
                        <div class="iteration-info">
                            <div class="iteration-indicator ${indicatorClass}"></div>
                            <span>${statusText}</span>
                        </div>
                        <div class="iteration-actions">
                            <button class="btn-recap-mini" onclick="recapCollect('${task.name}', ${iter.id})">Collect</button>
                            <button class="btn-recap-mini btn-recap-sim" onclick="recapSimulate('${task.name}', ${iter.id})">Sim</button>
                            ${hasData ? `<button class="btn-recap-mini" onclick="recapIterate('${task.name}', ${iter.id})">Iterate</button>` : ''}
                        </div>
                    </div>
                `;
                taskGroup.appendChild(iterRow);

                // Render data folders
                if (iter.data.length > 0) {
                    const dataList = document.createElement('div');
                    dataList.className = 'recap-data-list';
                    for (const dataObj of iter.data) {
                        const dataItem = document.createElement('div');
                        dataItem.className = 'recap-data-item';
                        dataItem.textContent = `📁 ${dataObj.id} (${dataObj.video_count} videos)`;
                        dataItem.onclick = () => loadRecapData(task.name, iter.id, dataObj.id);
                        dataList.appendChild(dataItem);
                    }
                    taskGroup.appendChild(dataList);
                }
            }

            recapTaskList.appendChild(taskGroup);
        }
    }

    window.recapSpecialize = async function (taskName) {
        try {
            const response = await fetch('/api/recap/specialize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_name: taskName })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                await fetchRecapState();
            } else {
                console.error('Specialize failed:', data.error);
            }
        } catch (error) {
            console.error('Specialize failed:', error);
        }
    };

    window.recapCollect = async function (taskName, iterId) {
        try {
            const response = await fetch('/api/recap/collect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_name: taskName, iter_id: iterId })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                await fetchRecapState();
            } else {
                console.error('Collect failed:', data.error);
            }
        } catch (error) {
            console.error('Collect failed:', error);
        }
    };

    window.recapSimulate = async function (taskName, iterId) {
        try {
            const response = await fetch('/api/recap/simulate_collection', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_name: taskName, iter_id: iterId })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                await fetchRecapState();
                if (activeVideo && activeVideo.filename.includes(taskName)) {
                    await fetchData();
                }
            }
        } catch (error) {
            console.error('Simulation failed:', error);
        }
    };

    window.recapIterate = async function (taskName, iterId) {
        try {
            const response = await fetch('/api/recap/iterate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_name: taskName, iter_id: iterId })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                await fetchRecapState();
            } else {
                console.error('Iterate failed:', data.error);
            }
        } catch (error) {
            console.error('Iterate failed:', error);
        }
    };

    async function loadPretrained() {
        loadingOverlay.classList.remove('hidden-element');
        loadingStatus.textContent = 'Loading Pretrained data...';
        loadingDetail.textContent = `pretrained_data`;

        try {
            const response = await fetch('/api/recap/load_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_pretrained: true })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                videos = [];
                labels = {};
                activeVideo = null;
                updateUIVisibility();
                videoList.innerHTML = '';
                carouselTrack.innerHTML = '';
                timelineContainer.innerHTML = '';
                loadingOverlay.dataset.recapLoading = 'true';
                checkConversionStatus();
            } else {
                console.error('Load failed:', data.error);
                loadingOverlay.classList.add('hidden-element');
            }
        } catch (error) {
            console.error('Load failed:', error);
            loadingOverlay.classList.add('hidden-element');
        }
    }
    window.loadPretrained = loadPretrained;

    async function loadRecapData(taskName, iterId, dataId) {
        loadingOverlay.classList.remove('hidden-element');
        loadingStatus.textContent = 'Loading RECAP data...';
        loadingDetail.textContent = `${taskName} / iter ${iterId} / ${dataId}`;

        try {
            const response = await fetch('/api/recap/load_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_name: taskName, iter_id: iterId, data_id: dataId })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                // Track the active folder
                const folderPath = `${taskName}/iter_${iterId}/${dataId}`;
                activeFolders.forEach(f => f.active = false);
                const existingFolder = activeFolders.find(f => f.path === folderPath);
                if (existingFolder) {
                    existingFolder.active = true;
                } else {
                    activeFolders.push({ path: folderPath, id: dataId, active: true });
                }
                updateActiveFoldersDisplay();
                updateFolderHighlighting();

                // Clear current state
                videos = [];
                labels = {};
                activeVideo = null;
                updateUIVisibility();
                videoList.innerHTML = '';
                carouselTrack.innerHTML = '';
                timelineContainer.innerHTML = '';
                // Check conversion status which will trigger fetchData when done
                loadingOverlay.dataset.recapLoading = 'true';
                checkConversionStatus();
            } else {
                console.error('Load failed:', data.error);
                loadingOverlay.classList.add('hidden-element');
            }
        } catch (error) {
            console.error('Load failed:', error);
            loadingOverlay.classList.add('hidden-element');
        }
    }

    // Finetune Policy button handler
    const policyTrainingModal = document.getElementById('policy-training-modal');
    const startPolicyTrainBtn = document.getElementById('btn-policy-start-train');
    const cancelPolicyTrainBtn = document.getElementById('btn-policy-cancel-train');
    const policyTrainingProgressView = document.getElementById('policy-training-progress-view');
    const policyTrainingSuccessView = document.getElementById('policy-training-success-view');
    const policyConfigOptions = document.querySelectorAll('.policy-config-selection .config-option');

    // Policy Training UI elements
    const policyTrainEpoch = document.getElementById('policy-train-epoch');
    const policyTrainLoss = document.getElementById('policy-train-loss');
    const policyTrainProgressBar = document.getElementById('policy-train-progress-bar');
    const policyTrainStepDetail = document.getElementById('policy-train-step-detail');
    const policySummaryConfig = document.getElementById('policy-summary-config');
    const policySummaryEpochs = document.getElementById('policy-summary-epochs');
    const policySummaryLoss = document.getElementById('policy-summary-loss');

    let selectedPolicyConfig = 'mock';

    btnFinetunePolicy.onclick = async () => {
        if (btnFinetunePolicy.disabled) return;

        policyTrainingModal.classList.remove('hidden-element');
        policyTrainingProgressView.classList.add('hidden-element');
        policyTrainingSuccessView.classList.add('hidden-element');

        startPolicyTrainBtn.disabled = false;
        startPolicyTrainBtn.textContent = 'Start Finetuning';
    };

    cancelPolicyTrainBtn.onclick = () => {
        policyTrainingModal.classList.add('hidden-element');
    };

    policyConfigOptions.forEach(opt => {
        opt.onclick = () => {
            policyConfigOptions.forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            selectedPolicyConfig = opt.dataset.config;
        };
    });

    startPolicyTrainBtn.onclick = async () => {
        try {
            startPolicyTrainBtn.disabled = true;
            startPolicyTrainBtn.textContent = 'Preparing...';

            const response = await fetch('/api/recap/finetune', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: selectedPolicyConfig })
            });
            const result = await response.json();
            if (result.error) {
                alert(result.error);
                policyTrainingModal.classList.add('hidden-element');
            } else {
                policyTrainingProgressView.classList.remove('hidden-element');
                startPolicyTrainBtn.disabled = true;
                startPolicyTrainBtn.textContent = 'Finetuning...';
            }
        } catch (error) {
            console.error('Policy fine-tuning failed:', error);
            policyTrainingModal.classList.add('hidden-element');
        }
    };

    // Pretrain button handler
    btnPretrain.onclick = async () => {
        try {
            const response = await fetch('/api/recap/pretrain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();
            if (data.status === 'ok') {
                await fetchRecapState();
            } else {
                console.error('Pretrain failed:', data.error);
            }
        } catch (error) {
            console.error('Pretrain failed:', error);
        }
    };

    // Initialize - check RECAP mode first
    async function initialize() {
        console.log("Starting initialization...");
        await fetchRecapState();
        console.log("RECAP state fetched:", recapState);

        // If in RECAP mode, don't start conversion polling
        // Just show the RECAP sidebar and wait for user to load data
        if (recapState && recapState.enabled) {
            console.log("RECAP mode active, explicit hide overlay");
            loadingOverlay.classList.add('hidden-element');
            updateUIVisibility();
            await fetchValueNetworks();
            await fetchData(); // Ensure videos/labels are loaded
            return;
        }

        // Otherwise, check conversion status for legacy folder mode
        console.log("Legacy mode: checking conversion status");
        checkConversionStatus();
    }

    initialize();

    // Training UI logic
    const trainValueBtn = document.getElementById('train-value-btn');
    const trainingModal = document.getElementById('training-modal');
    const startTrainBtn = document.getElementById('btn-start-train');
    const cancelTrainBtn = document.getElementById('btn-cancel-train');
    const simulateCollectionBtn = document.getElementById('btn-simulate-collection');
    const configOptions = document.querySelectorAll('.config-option');
    const trainingProgressView = document.getElementById('training-progress-view');
    const trainEpoch = document.getElementById('train-epoch');
    const trainLoss = document.getElementById('train-loss');
    const trainProgressBar = document.getElementById('train-progress-bar');
    const trainStepDetail = document.getElementById('train-step-detail');
    const trainingSuccessView = document.getElementById('training-success-view');
    const recapValueNetworks = document.getElementById('recap-value-networks');
    const summaryConfig = document.getElementById('summary-config');
    const summaryEpochs = document.getElementById('summary-epochs');
    const summaryLoss = document.getElementById('summary-loss');

    if (simulateCollectionBtn) {
        simulateCollectionBtn.onclick = async () => {
            // Pick a task from the state (e.g. first task not yet initialized with iter 0)
            const taskToSimulate = recapState.tasks.find(t => !t.iterations.some(it => it.id === 0))?.name || 'mock_task';
            try {
                const response = await fetch('/api/recap/simulate_collection', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ task_name: taskToSimulate })
                });
                const data = await response.json();
                if (data.status === 'ok') {
                    await fetchRecapState();
                }
            } catch (error) {
                console.error('Simulation failed:', error);
            }
        };
    }

    let selectedConfig = 'mock';
    let trainingSocket = null;
    let activeValueNetwork = null;

    function setupTrainingSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const socketUrl = `${protocol}//${window.location.host}/ws/training`;

        trainingSocket = new WebSocket(socketUrl);

        trainingSocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'training_update') {
                updateTrainingUI(data.state);
            }
        };

        trainingSocket.onerror = () => {
            console.error('Training WebSocket error');
        };

        trainingSocket.onclose = () => {
            setTimeout(setupTrainingSocket, 3000);
        };
    }

    function updateTrainingUI(state) {
        if (!state.is_training) {
            if (state.current_step > 0 && state.current_step >= state.total_steps) {
                // Show Success View based on active modal (simplistic check)
                if (!trainingModal.classList.contains('hidden-element')) {
                    // Value Network Success
                    trainingProgressView.classList.add('hidden-element');
                    trainingSuccessView.classList.remove('hidden-element');

                    summaryConfig.textContent = selectedConfig.toUpperCase();
                    summaryEpochs.textContent = state.current_epoch;
                    summaryLoss.textContent = state.last_loss.toFixed(6);

                    startTrainBtn.disabled = false;
                    startTrainBtn.textContent = 'Train Again';
                    fetchValueNetworks(); // Refresh list
                } else if (!policyTrainingModal.classList.contains('hidden-element')) {
                    // Policy Network Success
                    policyTrainingProgressView.classList.add('hidden-element');
                    policyTrainingSuccessView.classList.remove('hidden-element');

                    policySummaryConfig.textContent = selectedPolicyConfig.toUpperCase();
                    policySummaryEpochs.textContent = state.current_epoch;
                    policySummaryLoss.textContent = state.last_loss.toFixed(6);

                    startPolicyTrainBtn.disabled = false;
                    startPolicyTrainBtn.textContent = 'Finetune Again';
                }
            }
            return;
        }

        // Update Progress based on active modal
        if (!trainingModal.classList.contains('hidden-element')) {
            trainingProgressView.classList.remove('hidden-element');
            trainingSuccessView.classList.add('hidden-element');
            document.querySelector('.config-selection').classList.remove('hidden-element');
            startTrainBtn.disabled = true;
            startTrainBtn.textContent = 'Training...';

            trainEpoch.textContent = state.current_epoch;
            trainLoss.textContent = state.last_loss.toFixed(6);

            const progress = (state.current_step / state.total_steps) * 100;
            trainProgressBar.style.width = `${progress}%`;
            trainStepDetail.textContent = `Step ${state.current_step} / ${state.total_steps}`;

        } else if (!policyTrainingModal.classList.contains('hidden-element')) {
            policyTrainingProgressView.classList.remove('hidden-element');
            policyTrainingSuccessView.classList.add('hidden-element');

            startPolicyTrainBtn.disabled = true;
            startPolicyTrainBtn.textContent = 'Finetuning...';

            policyTrainEpoch.textContent = state.current_epoch;
            policyTrainLoss.textContent = state.last_loss.toFixed(6);

            const progress = (state.current_step / state.total_steps) * 100;
            policyTrainProgressBar.style.width = `${progress}%`;
            policyTrainStepDetail.textContent = `Step ${state.current_step} / ${state.total_steps}`;
        }
    }

    async function fetchValueNetworks() {
        try {
            const response = await fetch('/api/value/networks/list');
            const networks = await response.json();
            renderValueNetworks(networks);
        } catch (error) {
            console.error('Failed to fetch value networks:', error);
        }
    }

    function renderValueNetworks(networks) {
        recapValueNetworks.innerHTML = '';
        if (networks.length === 0) {
            recapValueNetworks.innerHTML = '<p class="tiny-label">No models trained yet.</p>';
        } else {
            networks.forEach(net => {
                const item = document.createElement('div');
                item.className = 'recap-value-item';
                if (activeValueNetwork === net.filename) {
                    item.classList.add('active');
                }

                const timestamp = net.timestamp.replace('_', ' ');
                item.innerHTML = `
                    <div class="value-item-header">
                        <span class="model-name">${net.config_name.toUpperCase()}</span>
                        <span class="model-loss">Loss: ${net.final_loss.toFixed(6)}</span>
                    </div>
                    <div class="value-item-meta">
                        <span>${timestamp}</span>
                        <span>${net.epochs} epochs</span>
                    </div>
                `;

                item.onclick = () => loadValueNetwork(net.filename);
                recapValueNetworks.appendChild(item);
            });
        }
    }

    async function loadValueNetwork(filename) {
        try {
            const response = await fetch('/api/value/networks/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            const data = await response.json();
            if (data.status === 'ok') {
                activeValueNetwork = filename;
                calcValueBtn.disabled = false;
                fetchValueNetworks(); // Refresh to update active state
            } else {
                alert('Load failed: ' + data.error);
            }
        } catch (error) {
            console.error('Load failed:', error);
        }
    }

    trainValueBtn.onclick = () => {
        trainingModal.classList.remove('hidden-element');
        // document.getElementById('training-modal-title').textContent = 'Train Value Network'; // Revert to static HTML
        trainingProgressView.classList.add('hidden-element');
        trainingSuccessView.classList.add('hidden-element');
        document.querySelector('.config-selection').classList.remove('hidden-element');
        startTrainBtn.disabled = false;
        startTrainBtn.textContent = 'Start Training';
    };

    const sidebarTrainBtn = document.getElementById('sidebar-train-btn');
    if (sidebarTrainBtn) {
        sidebarTrainBtn.onclick = trainValueBtn.onclick;
    }

    cancelTrainBtn.onclick = () => {
        trainingModal.classList.add('hidden-element');
    };

    configOptions.forEach(opt => {
        opt.onclick = () => {
            configOptions.forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            selectedConfig = opt.dataset.config;
        };
    });

    startTrainBtn.onclick = async () => {
        try {
            const response = await fetch('/api/value/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: selectedConfig })
            });
            const result = await response.json();
            if (result.error) {
                alert(result.error);
            } else {
                trainingProgressView.classList.remove('hidden-element');
                document.querySelector('.config-selection').classList.add('hidden-element');
            }
        } catch (error) {
            console.error('Training failed:', error);
        }
    };

    setupTrainingSocket();
    fetchValueNetworks();
});
