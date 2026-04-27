/* ═══════════════════════════════════════════════════════════════════════════
   YouTube Video Q&A — Application Logic
   ═══════════════════════════════════════════════════════════════════════════ */

(() => {
    "use strict";

    // ── DOM References ────────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const sectionInput      = $("#section-input");
    const sectionProcessing = $("#section-processing");
    const sectionChat       = $("#section-chat");

    const urlInput   = $("#youtube-url");
    const btnClear   = $("#btn-clear");
    const btnProcess = $("#btn-process");

    const processingTitle = $("#processing-title");
    const videoPreview    = $("#video-preview");
    const videoThumb      = $("#video-thumbnail");
    const videoTitle      = $("#video-title");
    const videoChannel    = $("#video-channel");
    const videoDuration   = $("#video-duration");
    const progressFill    = $("#progress-fill");
    const progressStatus  = $("#progress-status");
    const progressPct     = $("#progress-pct");

    const sidebarThumb   = $("#sidebar-thumb");
    const sidebarTitle   = $("#sidebar-title");
    const sidebarChannel = $("#sidebar-channel");
    const transcriptPrev = $("#transcript-preview");

    const messagesEl    = $("#messages");
    const questionInput = $("#question-input");
    const btnAsk        = $("#btn-ask");
    const btnNewVideo   = $("#btn-new-video");

    const toast        = $("#toast");
    const toastMessage = $("#toast-message");

    // ── State ─────────────────────────────────────────────────────────────
    let currentVideoId = null;
    let currentJobId   = null;
    let pollingTimer   = null;

    // ── Pipeline step mapping ─────────────────────────────────────────────
    const STATUS_LABELS = {
        queued:            "Initializing pipeline...",
        fetching_metadata: "Fetching video metadata...",
        downloading:       "Downloading audio track...",
        transcribing:      "Transcribing with Whisper AI...",
        embedding:         "Chunking & embedding into Pinecone...",
        ready:             "All set! Launching chat...",
        error:             "An error occurred.",
    };

    const STEP_ORDER = [
        "fetching_metadata",
        "downloading",
        "transcribing",
        "embedding",
        "ready",
    ];

    // ── Helpers ───────────────────────────────────────────────────────────

    function showToast(msg, duration = 4000) {
        toastMessage.textContent = msg;
        toast.style.display = "flex";
        toast.style.animation = "none";
        // Force reflow
        toast.offsetHeight;
        toast.style.animation = "";
        clearTimeout(toast._timer);
        toast._timer = setTimeout(() => {
            toast.style.display = "none";
        }, duration);
    }

    function formatDuration(seconds) {
        if (!seconds) return "";
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${h}h ${m}m ${s}s`;
        return `${m}m ${s}s`;
    }

    function showSection(section) {
        [sectionInput, sectionProcessing, sectionChat].forEach((s) => {
            s.style.display = "none";
        });
        section.style.display = "";
        section.style.animation = "none";
        section.offsetHeight;
        section.style.animation = "";
    }

    function autoResizeTextarea(el) {
        el.style.height = "auto";
        el.style.height = Math.min(el.scrollHeight, 120) + "px";
    }

    // ── URL Input Handlers ────────────────────────────────────────────────

    urlInput.addEventListener("input", () => {
        btnClear.style.display = urlInput.value.trim() ? "block" : "none";
    });

    btnClear.addEventListener("click", () => {
        urlInput.value = "";
        btnClear.style.display = "none";
        urlInput.focus();
    });

    urlInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            btnProcess.click();
        }
    });

    // ── Process Video ─────────────────────────────────────────────────────

    btnProcess.addEventListener("click", async () => {
        const url = urlInput.value.trim();
        if (!url) {
            showToast("Please paste a YouTube URL first.");
            urlInput.focus();
            return;
        }

        if (!url.includes("youtube.com") && !url.includes("youtu.be")) {
            showToast("That doesn't look like a valid YouTube URL.");
            return;
        }

        btnProcess.disabled = true;

        try {
            const res = await fetch("/api/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url }),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Failed to start processing.");
            }

            const data = await res.json();
            currentJobId = data.job_id;

            // If already processed, go straight to chat
            if (data.message.includes("already processed")) {
                currentVideoId = data.job_id;
                startChat();
                return;
            }

            // Show processing UI
            showSection(sectionProcessing);
            resetProcessingUI();
            startPolling();
        } catch (err) {
            showToast(err.message);
        } finally {
            btnProcess.disabled = false;
        }
    });

    // ── Polling ───────────────────────────────────────────────────────────

    function resetProcessingUI() {
        progressFill.style.width = "0%";
        progressPct.textContent = "0%";
        progressStatus.textContent = "Initializing...";
        videoPreview.style.display = "none";
        $$(".progress-step").forEach((s) => {
            s.classList.remove("active", "done");
        });

        // Reset pipeline steps
        $$(".pipeline__step").forEach((s) => {
            s.classList.remove("active", "done");
        });
        $$(".pipeline__connector").forEach((c) => {
            c.classList.remove("active");
        });
    }

    function startPolling() {
        if (pollingTimer) clearInterval(pollingTimer);
        pollingTimer = setInterval(pollStatus, 1500);
        pollStatus(); // Immediate first poll
    }

    async function pollStatus() {
        if (!currentJobId) return;

        try {
            const res = await fetch(`/api/status/${currentJobId}`);
            if (!res.ok) return;

            const data = await res.json();

            // Update progress
            progressFill.style.width = data.progress + "%";
            progressPct.textContent = data.progress + "%";
            progressStatus.textContent = STATUS_LABELS[data.status] || data.status;

            // Update step indicators
            const currentIdx = STEP_ORDER.indexOf(data.status);
            $$(".progress-step").forEach((el) => {
                const step = el.dataset.pstep;
                const stepIdx = STEP_ORDER.indexOf(step);
                el.classList.remove("active", "done");
                if (stepIdx < currentIdx) el.classList.add("done");
                if (stepIdx === currentIdx) el.classList.add("active");
            });

            // Update pipeline visualization
            const pipelineMap = {
                downloading: "download",
                transcribing: "transcribe",
                embedding: "embed",
                ready: "ready",
            };
            const pipelineSteps = $$(".pipeline__step");
            const connectors = $$(".pipeline__connector");

            pipelineSteps.forEach((el, i) => {
                const step = el.dataset.step;
                el.classList.remove("active", "done");
                const stepKeys = ["download", "transcribe", "embed", "ready"];
                const sIdx = stepKeys.indexOf(step);
                const cIdx = stepKeys.indexOf(pipelineMap[data.status] || "");
                if (sIdx < cIdx) el.classList.add("done");
                if (sIdx === cIdx) el.classList.add("active");
            });

            connectors.forEach((c, i) => {
                const stepKeys = ["download", "transcribe", "embed", "ready"];
                const cIdx = stepKeys.indexOf(pipelineMap[data.status] || "");
                c.classList.toggle("active", i < cIdx);
            });

            // Show video preview
            if (data.metadata && !videoPreview.style.display !== "") {
                videoThumb.src = data.metadata.thumbnail || "";
                videoTitle.textContent = data.metadata.title || "";
                videoChannel.textContent = data.metadata.channel || "";
                videoDuration.textContent = formatDuration(data.metadata.duration);
                videoPreview.style.display = "flex";
            }

            if (data.video_id) {
                currentVideoId = data.video_id;
            }

            // Ready!
            if (data.status === "ready") {
                clearInterval(pollingTimer);
                processingTitle.textContent = "Processing complete!";
                $(".processing-spinner").style.display = "none";
                setTimeout(startChat, 1200);
            }

            // Error
            if (data.status === "error") {
                clearInterval(pollingTimer);
                processingTitle.textContent = "Processing failed";
                $(".processing-spinner").style.display = "none";
                showToast(data.error || "Something went wrong during processing.");
            }
        } catch (err) {
            console.error("Polling error:", err);
        }
    }

    // ── Chat ──────────────────────────────────────────────────────────────

    function startChat() {
        showSection(sectionChat);

        // Fill sidebar
        if (videoThumb.src) sidebarThumb.src = videoThumb.src;
        if (videoTitle.textContent) sidebarTitle.textContent = videoTitle.textContent;
        if (videoChannel.textContent) sidebarChannel.textContent = videoChannel.textContent;

        // Load transcript preview
        loadTranscriptPreview();

        // Clear previous messages except the system welcome
        const msgs = messagesEl.querySelectorAll(".message:not(.message--system)");
        msgs.forEach((m) => m.remove());

        questionInput.focus();
    }

    async function loadTranscriptPreview() {
        if (!currentVideoId) return;
        try {
            const res = await fetch(`/api/transcript/${currentVideoId}`);
            if (res.ok) {
                const data = await res.json();
                const preview = data.transcript ? data.transcript.substring(0, 600) : "";
                transcriptPrev.innerHTML = `<p class="transcript-preview__text">${preview}${preview.length >= 600 ? "..." : ""}</p>`;
            }
        } catch (err) {
            console.error("Transcript load error:", err);
        }
    }

    // Send message
    async function sendQuestion() {
        const q = questionInput.value.trim();
        if (!q || !currentVideoId) return;

        // Disable input
        btnAsk.disabled = true;
        questionInput.value = "";
        autoResizeTextarea(questionInput);

        // Add user message
        appendMessage("user", q);

        // Add typing indicator
        const typingEl = appendTyping();

        try {
            const res = await fetch("/api/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    video_id: currentVideoId,
                    question: q,
                }),
            });

            typingEl.remove();

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Failed to get a response.");
            }

            const data = await res.json();
            appendMessage("ai", data.answer, data.sources);
        } catch (err) {
            typingEl.remove();
            appendMessage("ai", "Sorry, I encountered an error: " + err.message);
        } finally {
            btnAsk.disabled = false;
            questionInput.focus();
        }
    }

    function appendMessage(type, text, sources = []) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message message--${type}`;

        const avatarIcon =
            type === "user"
                ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`
                : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`;

        let sourcesHTML = "";
        if (sources && sources.length > 0) {
            const items = sources
                .slice(0, 3)
                .map(
                    (s) =>
                        `<div class="message__source-item">${escapeHTML(s.substring(0, 150))}...</div>`
                )
                .join("");
            sourcesHTML = `
                <div class="message__sources">
                    <div class="message__sources-label">📎 Sources (${sources.length} chunks)</div>
                    ${items}
                </div>`;
        }

        msgDiv.innerHTML = `
            <div class="message__avatar message__avatar--${type}">
                ${avatarIcon}
            </div>
            <div class="message__content">
                <p>${formatMessageText(text)}</p>
                ${sourcesHTML}
            </div>`;

        messagesEl.appendChild(msgDiv);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return msgDiv;
    }

    function appendTyping() {
        const div = document.createElement("div");
        div.className = "message message--ai";
        div.innerHTML = `
            <div class="message__avatar message__avatar--ai">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                </svg>
            </div>
            <div class="message__content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>`;
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return div;
    }

    function escapeHTML(str) {
        const d = document.createElement("div");
        d.textContent = str;
        return d.innerHTML;
    }

    function formatMessageText(text) {
        // Basic markdown-like formatting
        return escapeHTML(text)
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.*?)\*/g, "<em>$1</em>")
            .replace(/`(.*?)`/g, '<code style="background:var(--bg-primary);padding:2px 6px;border-radius:4px;font-family:var(--font-mono);font-size:0.85em;">$1</code>')
            .replace(/\n/g, "<br>");
    }

    // ── Event Listeners ───────────────────────────────────────────────────

    btnAsk.addEventListener("click", sendQuestion);

    questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });

    questionInput.addEventListener("input", () => {
        autoResizeTextarea(questionInput);
    });

    // Suggested questions
    $$(".sq-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            questionInput.value = btn.dataset.q;
            autoResizeTextarea(questionInput);
            sendQuestion();
        });
    });

    // New video
    btnNewVideo.addEventListener("click", () => {
        currentVideoId = null;
        currentJobId = null;
        urlInput.value = "";
        btnClear.style.display = "none";
        showSection(sectionInput);
        urlInput.focus();
    });

    // ── Initialization ────────────────────────────────────────────────────
    urlInput.focus();
})();
