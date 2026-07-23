/* CareGPT UI behaviour. No external JavaScript dependencies are required. */
(function () {
  "use strict";

  const body = document.body;
  const themeButton = document.querySelector("[data-theme-toggle]");
  const savedTheme = window.localStorage.getItem("caregpt-theme");

  if (savedTheme === "dark") {
    body.classList.add("theme-dark");
  }

  themeButton?.addEventListener("click", () => {
    const isDark = body.classList.toggle("theme-dark");
    window.localStorage.setItem("caregpt-theme", isDark ? "dark" : "light");
    themeButton.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
  });

  const header = document.querySelector("[data-header]");
  const refreshHeader = () => header?.classList.toggle("is-scrolled", window.scrollY > 10);
  refreshHeader();
  window.addEventListener("scroll", refreshHeader, { passive: true });

  const riskForm = document.querySelector("[data-risk-form]");
  riskForm?.addEventListener("submit", (event) => {
    const inputs = Array.from(riskForm.querySelectorAll("input[required], select[required]"));
    const invalid = inputs.find((input) => !input.checkValidity());
    inputs.forEach((input) => input.toggleAttribute("aria-invalid", !input.checkValidity()));
    if (invalid) {
      event.preventDefault();
      invalid.focus();
      invalid.reportValidity();
    }
  });

  riskForm?.addEventListener("input", (event) => {
    if (event.target.matches("input, select")) {
      event.target.removeAttribute("aria-invalid");
    }
  });

  document.querySelectorAll("[data-print]").forEach((button) => {
    button.addEventListener("click", () => window.print());
  });

  const chatForm = document.querySelector("[data-chat-form]");
  if (!chatForm) return;

  const chatInput = chatForm.querySelector("[data-chat-input]");
  const chatLog = document.querySelector("[data-chat-log]");
  const sendButton = chatForm.querySelector("button[type='submit']");
  const suggestions = document.querySelectorAll("[data-suggestion]");

  function scrollChatToEnd() {
    chatLog.scrollTo({ top: chatLog.scrollHeight, behavior: "smooth" });
  }

  function autoResize() {
    chatInput.style.height = "auto";
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 110)}px`;
  }

  function makeMessage(text, type) {
    const article = document.createElement("article");
    article.className = `message message-${type}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = type === "user" ? "◌" : "✦";

    const content = document.createElement("div");
    content.className = "message-content";
    const paragraph = document.createElement("p");
    paragraph.textContent = text;
    content.appendChild(paragraph);
    article.append(avatar, content);
    return article;
  }

  function showTyping() {
    const article = document.createElement("article");
    article.className = "message message-assistant message-typing";
    article.dataset.typing = "true";
    article.innerHTML = '<div class="message-avatar" aria-hidden="true">✦</div><div class="message-content"><span class="typing-dots"><i></i><i></i><i></i></span></div>';
    chatLog.appendChild(article);
    scrollChatToEnd();
    return article;
  }

  async function submitMessage(message) {
    const trimmed = message.trim();
    if (!trimmed || sendButton.disabled) return;

    chatLog.appendChild(makeMessage(trimmed, "user"));
    chatInput.value = "";
    autoResize();
    scrollChatToEnd();
    sendButton.disabled = true;
    const typing = showTyping();

    const formData = new FormData(chatForm);
    formData.set("message", trimmed);

    try {
      const response = await fetch(chatForm.action, {
        method: "POST",
        body: formData,
        headers: { Accept: "application/json", "X-Requested-With": "XMLHttpRequest" },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.reply) {
        throw new Error(data.description || "The assistant is unavailable right now.");
      }
      typing.remove();
      chatLog.appendChild(makeMessage(data.reply, "assistant"));
    } catch (error) {
      typing.remove();
      chatLog.appendChild(makeMessage("I’m sorry, I couldn’t send that just now. Please try again. If you have urgent symptoms, seek emergency care immediately.", "assistant"));
      console.warn("CareGPT chat request failed:", error);
    } finally {
      sendButton.disabled = false;
      scrollChatToEnd();
      chatInput.focus();
    }
  }

  chatInput.addEventListener("input", autoResize);
  chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      chatForm.requestSubmit();
    }
  });

  chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitMessage(chatInput.value);
  });

  suggestions.forEach((button) => {
    button.addEventListener("click", () => submitMessage(button.dataset.suggestion || ""));
  });
}());
