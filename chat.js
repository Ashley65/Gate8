/* ==========================================================================
   GATE8 CHAT INTERFACE CONTROLLER
   Gemini & Claude-inspired Chat Experience with Kokoro TTS Integration
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Elements Selection
    const sidebar = document.getElementById('sidebar');
    const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
    const closeSidebarBtn = document.getElementById('closeSidebarBtn');
    const newChatBtn = document.getElementById('newChatBtn');

    const modelSelectBtn = document.getElementById('modelSelectBtn');
    const modelDropdownMenu = document.getElementById('modelDropdownMenu');
    const selectedModelLabel = document.getElementById('selectedModelLabel');
    const currentModelBadge = document.getElementById('currentModelBadge');
    const modelOptions = document.querySelectorAll('.model-option');

    const welcomeHero = document.getElementById('welcomeHero');
    const messagesFeed = document.getElementById('messagesFeed');
    const chatMessagesContainer = document.getElementById('chatMessagesContainer');
    const presetCards = document.querySelectorAll('.preset-card');

    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const autoTTSToggle = document.getElementById('autoTTSToggle');
    const micBtn = document.getElementById('micBtn');
    const speakSampleBtn = document.getElementById('speakSampleBtn');
    const waveformContainer = document.getElementById('waveformContainer');

    const chatSettingsBtn = document.getElementById('chatSettingsBtn');
    const settingsModal = document.getElementById('settingsModal');
    const closeSettingsBtn = document.getElementById('closeSettingsBtn');
    const saveSettingsBtn = document.getElementById('saveSettingsBtn');
    const tempSlider = document.getElementById('tempSlider');
    const tempValue = document.getElementById('tempValue');

    // Sidebar Toggles
    if (toggleSidebarBtn) {
        toggleSidebarBtn.addEventListener('click', () => {
            sidebar.classList.toggle('active');
        });
    }
    if (closeSidebarBtn) {
        closeSidebarBtn.addEventListener('click', () => {
            sidebar.classList.remove('active');
        });
    }

    // Model Selector Dropdown
    modelSelectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        modelDropdownMenu.classList.toggle('show');
    });

    document.addEventListener('click', () => {
        modelDropdownMenu.classList.remove('show');
    });

    modelOptions.forEach(opt => {
        opt.addEventListener('click', () => {
            modelOptions.forEach(o => o.classList.remove('active'));
            opt.classList.add('active');
            const modelName = opt.querySelector('.model-name').textContent;
            selectedModelLabel.textContent = modelName;
            currentModelBadge.textContent = modelName;
            modelDropdownMenu.classList.remove('show');
        });
    });

    // Preset Cards Click
    presetCards.forEach(card => {
        card.addEventListener('click', () => {
            const promptText = card.getAttribute('data-prompt');
            chatInput.value = promptText;
            chatInput.focus();
            autoResizeTextarea();
        });
    });

    // Textarea Auto Resize
    function autoResizeTextarea() {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
    }
    chatInput.addEventListener('input', autoResizeTextarea);

    // Keyboard Shortcuts (Enter / Shift+Enter / Ctrl+K)
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            startNewChat();
        }
    });

    newChatBtn.addEventListener('click', startNewChat);

    function startNewChat() {
        welcomeHero.classList.remove('hidden');
        messagesFeed.innerHTML = '';
        chatInput.value = '';
        autoResizeTextarea();
    }

    // Send Message Logic
    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Hide welcome hero if visible
        if (welcomeHero) welcomeHero.classList.add('hidden');

        // Render User Message
        appendUserMessage(text);
        chatInput.value = '';
        autoResizeTextarea();

        // Render Assistant Loading State
        const loadingRow = appendAssistantLoading();
        scrollToBottom();

        // Call Gate8 API
        const activeModel = selectedModelLabel.textContent;
        const response = await Gate8API.sendChatCompletion(text, activeModel);

        // Remove loading and render Assistant Message
        loadingRow.remove();
        const aiMsgText = response.choices[0].message.content;
        appendAssistantMessage(aiMsgText, activeModel);
        scrollToBottom();

        // Auto-Speak TTS if enabled
        if (autoTTSToggle && autoTTSToggle.checked) {
            triggerTTSPlayback(aiMsgText);
        }
    }

    sendBtn.addEventListener('click', sendMessage);

    function appendUserMessage(text) {
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const row = document.createElement('div');
        row.className = 'message-row user-row';
        row.innerHTML = `
            <div class="avatar user-avatar">
                <i class="fa-solid fa-user"></i>
            </div>
            <div class="message-body">
                <div class="message-meta">
                    <span class="sender-name">You</span>
                    <span class="message-time">${timeStr}</span>
                </div>
                <div class="message-text">${escapeHTML(text)}</div>
            </div>
        `;
        messagesFeed.appendChild(row);
    }

    function appendAssistantLoading() {
        const row = document.createElement('div');
        row.className = 'message-row assistant-row loading-row';
        row.innerHTML = `
            <div class="avatar ai-avatar">
                <i class="fa-solid fa-sparkles"></i>
            </div>
            <div class="message-body">
                <div class="message-meta">
                    <span class="sender-name">Gate8 Core</span>
                    <span class="message-time">Generating...</span>
                </div>
                <div class="message-text">
                    <span class="dot-typing">Processing speech & grammar polish...</span>
                </div>
            </div>
        `;
        messagesFeed.appendChild(row);
        return row;
    }

    function appendAssistantMessage(contentHTML, modelName) {
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const row = document.createElement('div');
        row.className = 'message-row assistant-row';
        row.innerHTML = `
            <div class="avatar ai-avatar">
                <i class="fa-solid fa-sparkles"></i>
            </div>
            <div class="message-body">
                <div class="message-meta">
                    <span class="sender-name">Gate8 Core (${modelName})</span>
                    <span class="message-time">${timeStr}</span>
                </div>
                <div class="message-text markdown-content">
                    ${formatMarkdown(contentHTML)}
                </div>
                <div class="audio-narration-bar">
                    <button class="btn btn-secondary speak-btn">
                        <i class="fa-solid fa-play"></i>
                        <span>Speak Response (Kokoro TTS)</span>
                    </button>
                    <div class="audio-waveform-visualizer hidden">
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <span class="audio-duration">0:14 / 0:14</span>
                    </div>
                </div>
                <div class="message-actions-toolbar">
                    <button class="action-btn" title="Copy Text"><i class="fa-regular fa-copy"></i></button>
                    <button class="action-btn" title="Export WAV to R2"><i class="fa-solid fa-cloud-arrow-up"></i></button>
                    <button class="action-btn" title="Regenerate Response"><i class="fa-solid fa-arrows-rotate"></i></button>
                </div>
            </div>
        `;

        // Bind TTS speak button on message
        const speakBtn = row.querySelector('.speak-btn');
        const viz = row.querySelector('.audio-waveform-visualizer');
        speakBtn.addEventListener('click', () => {
            triggerTTSVisualizer(speakBtn, viz);
        });

        messagesFeed.appendChild(row);
    }

    // Kokoro TTS Visualizer Player Simulation
    if (speakSampleBtn && waveformContainer) {
        speakSampleBtn.addEventListener('click', () => {
            triggerTTSVisualizer(speakSampleBtn, waveformContainer);
        });
    }

    function triggerTTSVisualizer(btn, viz) {
        viz.classList.remove('hidden');
        btn.querySelector('span').textContent = 'Synthesizing Audio...';
        btn.querySelector('i').className = 'fa-solid fa-circle-notch fa-spin';

        setTimeout(() => {
            btn.querySelector('span').textContent = 'Playing Kokoro Stream';
            btn.querySelector('i').className = 'fa-solid fa-volume-high';
            
            setTimeout(() => {
                btn.querySelector('span').textContent = 'Speak Response (Kokoro TTS)';
                btn.querySelector('i').className = 'fa-solid fa-play';
                viz.classList.add('hidden');
            }, 5000);
        }, 1200);
    }

    function triggerTTSPlayback(text) {
        const lastMsgBtn = messagesFeed.querySelector('.assistant-row:last-child .speak-btn');
        const lastViz = messagesFeed.querySelector('.assistant-row:last-child .audio-waveform-visualizer');
        if (lastMsgBtn && lastViz) {
            triggerTTSVisualizer(lastMsgBtn, lastViz);
        }
    }

    // Scroll Helper
    function scrollToBottom() {
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    }

    // Helpers
    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
        );
    }

    function formatMarkdown(str) {
        return str
            .replace(/\n\n/g, '</p><p>')
            .replace(/> "(.*?)"/g, '<blockquote>"$1"</blockquote>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }

    // Modal Settings Toggle
    chatSettingsBtn.addEventListener('click', () => settingsModal.classList.remove('hidden'));
    closeSettingsBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));
    saveSettingsBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));

    tempSlider.addEventListener('input', () => {
        tempValue.textContent = tempSlider.value;
    });
});
