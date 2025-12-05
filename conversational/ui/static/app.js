/**
 * AI Data Pipeline - Frontend Application
 * 
 * Handles:
 * - Chat communication with backend
 * - Real-time pipeline state polling
 * - Stage output display
 * - UI interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    // ========================================================================
    // DOM ELEMENTS
    // ========================================================================

    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const clearChatBtn = document.getElementById('clear-chat');
    const detailsContent = document.getElementById('details-content');
    const detailsTitle = document.getElementById('details-title');
    const refreshBtn = document.getElementById('refresh-btn');
    const expandBtn = document.getElementById('expand-btn');
    const statusIndicator = document.getElementById('status-indicator');
    const modal = document.getElementById('expand-modal');
    const modalClose = document.getElementById('modal-close');
    const modalBody = document.getElementById('modal-body');
    const modalTitle = document.getElementById('modal-title');

    // Stage chips in header
    const stageChips = document.querySelectorAll('.stage-chip');
    const quickBtns = document.querySelectorAll('.quick-btn');

    // ========================================================================
    // STATE
    // ========================================================================

    let currentStage = null;
    let isPolling = false;
    let pollInterval = null;
    let currentStageData = null;

    // ========================================================================
    // CHAT FUNCTIONALITY
    // ========================================================================

    function createMessageElement(role, content) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? '👤' : '🤖';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        // Parse content - handle newlines and basic formatting
        let formatted = content
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/`(.*?)`/g, '<code>$1</code>');

        contentDiv.innerHTML = `<p>${formatted}</p>`;

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(contentDiv);

        return msgDiv;
    }

    function addMessage(role, content) {
        const msgElement = createMessageElement(role, content);
        chatMessages.appendChild(msgElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function sendMessage(text = null) {
        const message = text || userInput.value.trim();
        if (!message) return;

        addMessage('user', message);
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            addMessage('assistant', data.response);

            if (data.pipeline_started) {
                startPolling();
                updateStatusIndicator(true);
            }

        } catch (error) {
            console.error('Chat error:', error);
            addMessage('assistant', '❌ Sorry, I encountered an error. Please try again.');
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    // Event listeners for chat
    sendBtn.addEventListener('click', () => sendMessage());

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    clearChatBtn.addEventListener('click', () => {
        chatMessages.innerHTML = '';
        addMessage('assistant', '💬 Chat cleared. How can I help you?');
    });

    // Quick action buttons
    quickBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            sendMessage(btn.dataset.message);
        });
    });

    // ========================================================================
    // STATE POLLING
    // ========================================================================

    async function fetchState() {
        try {
            const response = await fetch('/api/state');
            if (!response.ok) return null;
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch state:', error);
            return null;
        }
    }

    function updateStageIndicators(state) {
        if (!state || !state.stages) return;

        const stages = state.stages;

        stageChips.forEach(chip => {
            const stageName = chip.dataset.stage;
            const stageData = stages[stageName];

            // Reset classes
            chip.classList.remove('pending', 'running', 'completed', 'failed');

            if (stageData) {
                const status = stageData.status || 'pending';
                chip.classList.add(status);

                // Update number icon
                const numEl = chip.querySelector('.stage-number');
                if (status === 'completed') {
                    numEl.textContent = '✓';
                } else if (status === 'failed') {
                    numEl.textContent = '✗';
                } else if (status === 'running') {
                    numEl.textContent = '⟳';
                } else {
                    // Restore original
                    const originalText = getStageNumber(stageName);
                    numEl.textContent = originalText;
                }
            } else {
                chip.classList.add('pending');
            }
        });
    }

    function getStageNumber(stageName) {
        const map = {
            'stage1': '1',
            'stage2': '2',
            'stage3': '3',
            'stage3b': '3B',
            'stage3_5a': '3.5A',
            'stage3_5b': '3.5B',
            'stage4': '4',
            'stage5': '5'
        };
        return map[stageName] || '?';
    }

    function updateStatusIndicator(isRunning) {
        statusIndicator.classList.toggle('running', isRunning);
        statusIndicator.querySelector('.status-text').textContent =
            isRunning ? 'Running...' : 'Ready';
    }

    function startPolling() {
        if (isPolling) return;
        isPolling = true;

        const poll = async () => {
            const state = await fetchState();

            if (state) {
                updateStageIndicators(state);

                const isRunning = state.is_running;
                updateStatusIndicator(isRunning);

                // If current stage is selected, refresh its details
                if (currentStage) {
                    loadStageDetails(currentStage, false);
                }

                if (!isRunning) {
                    stopPolling();
                    return;
                }
            }

            pollInterval = setTimeout(poll, 2000);
        };

        poll();
    }

    function stopPolling() {
        isPolling = false;
        if (pollInterval) {
            clearTimeout(pollInterval);
            pollInterval = null;
        }
    }

    // ========================================================================
    // STAGE DETAILS
    // ========================================================================

    async function loadStageDetails(stageName, showLoading = true) {
        currentStage = stageName;

        // Update active state in header
        stageChips.forEach(chip => {
            chip.classList.toggle('active', chip.dataset.stage === stageName);
        });

        const stageTitle = getStageTitle(stageName);
        detailsTitle.textContent = `📁 ${stageTitle}`;

        if (showLoading) {
            detailsContent.innerHTML = '<div class="loading">Loading stage details...</div>';
        }

        try {
            const response = await fetch(`/api/stage/${stageName}`);

            if (!response.ok) {
                detailsContent.innerHTML = `
                    <div class="stage-output">
                        <div class="output-header">
                            <h3>📋 ${stageTitle}</h3>
                            <span class="status-badge pending">Pending</span>
                        </div>
                        <div class="output-body">
                            <p style="color: var(--text-secondary);">This stage has not been executed yet.</p>
                        </div>
                    </div>
                `;
                return;
            }

            const data = await response.json();
            currentStageData = data;
            renderStageDetails(data, stageTitle);

        } catch (error) {
            console.error('Failed to load stage details:', error);
            detailsContent.innerHTML = `
                <div class="error-message">
                    Failed to load stage details: ${error.message}
                </div>
            `;
        }
    }

    function getStageTitle(stageName) {
        const titles = {
            'stage1': 'Data Analysis',
            'stage2': 'Task Proposals',
            'stage3': 'Execution Plan',
            'stage3b': 'Data Preparation',
            'stage3_5a': 'Method Selection',
            'stage3_5b': 'Benchmarking',
            'stage4': 'Execution Results',
            'stage5': 'Visualizations'
        };
        return titles[stageName] || stageName;
    }

    function renderStageDetails(data, title) {
        const status = data.status || 'pending';
        const output = data.output;

        let outputHtml = '';

        if (output) {
            // Format the output based on stage type
            if (typeof output === 'object') {
                outputHtml = `<pre>${syntaxHighlight(output)}</pre>`;
            } else {
                outputHtml = `<pre>${escapeHtml(String(output))}</pre>`;
            }
        } else {
            outputHtml = '<p style="color: var(--text-secondary);">No output available yet.</p>';
        }

        detailsContent.innerHTML = `
            <div class="stage-output">
                <div class="output-header">
                    <h3>📋 ${title}</h3>
                    <span class="status-badge ${status}">${status}</span>
                </div>
                <div class="output-body">
                    ${outputHtml}
                </div>
            </div>
        `;
    }

    // ========================================================================
    // JSON SYNTAX HIGHLIGHTING
    // ========================================================================

    function syntaxHighlight(obj) {
        let json = JSON.stringify(obj, null, 2);
        json = escapeHtml(json);

        return json.replace(
            /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
            (match) => {
                let cls = 'json-number';
                if (/^"/.test(match)) {
                    if (/:$/.test(match)) {
                        cls = 'json-key';
                    } else {
                        cls = 'json-string';
                    }
                } else if (/true|false/.test(match)) {
                    cls = 'json-boolean';
                } else if (/null/.test(match)) {
                    cls = 'json-null';
                }
                return `<span class="${cls}">${match}</span>`;
            }
        );
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // ========================================================================
    // STAGE CHIP CLICK HANDLERS
    // ========================================================================

    stageChips.forEach(chip => {
        chip.addEventListener('click', () => {
            loadStageDetails(chip.dataset.stage);
        });
    });

    // ========================================================================
    // REFRESH BUTTON
    // ========================================================================

    refreshBtn.addEventListener('click', async () => {
        const state = await fetchState();
        if (state) {
            updateStageIndicators(state);
        }

        if (currentStage) {
            loadStageDetails(currentStage);
        }
    });

    // ========================================================================
    // EXPAND MODAL
    // ========================================================================

    expandBtn.addEventListener('click', () => {
        if (currentStageData) {
            modalTitle.textContent = getStageTitle(currentStage);
            modalBody.innerHTML = `<pre>${syntaxHighlight(currentStageData)}</pre>`;
            modal.classList.add('active');
        }
    });

    modalClose.addEventListener('click', () => {
        modal.classList.remove('active');
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            modal.classList.remove('active');
        }
    });

    // ========================================================================
    // INITIAL LOAD
    // ========================================================================

    async function initialize() {
        const state = await fetchState();
        if (state) {
            updateStageIndicators(state);
            updateStatusIndicator(state.is_running);

            if (state.is_running) {
                startPolling();
            }
        }
    }

    initialize();
});
