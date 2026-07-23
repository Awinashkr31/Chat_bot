(function(){
  const uidArea = document.getElementById('uidArea');
  const uidInput = document.getElementById('uidInput');
  const uidSubmit = document.getElementById('uidSubmit');
  const uidSkip = document.getElementById('uidSkip');
  const uidMsg = document.getElementById('uidMsg');
  const chat = document.getElementById('chat');
  const messageInput = document.getElementById('messageInput');
  const language = document.getElementById('language');
  const sendBtn = document.getElementById('sendBtn');
  const voiceBtn = document.getElementById('voiceBtn');
  const resetBtn = document.getElementById('resetBtn');
  const muteBtn = document.getElementById('muteBtn');

  let USER_UID = localStorage.getItem('mca_uid') || '';
  let MUTED = localStorage.getItem('mca_muted') === '1';
  let SESSION_ID = localStorage.getItem('mca_session_id');
  if (!SESSION_ID) {
    SESSION_ID = Math.random().toString(36).substring(2, 15);
    localStorage.setItem('mca_session_id', SESSION_ID);
  }
  let isSpeaking = false;
  
  chat.addEventListener('click', () => {
    if (isSpeaking && window.speechSynthesis) window.speechSynthesis.cancel();
  });

  function setMuted(v) {
    MUTED = !!v;
    localStorage.setItem('mca_muted', MUTED ? '1' : '0');
    muteBtn.textContent = MUTED ? 'Unmute' : 'Mute';
    if (MUTED && window.speechSynthesis) window.speechSynthesis.cancel();
  }
  setMuted(MUTED);

  function saveMessageToHistory(cls, text, isClarify=false) {
    let history = JSON.parse(localStorage.getItem('mca_history') || '[]');
    history.push({ cls, text, isClarify });
    if (history.length > 50) history = history.slice(-50);
    localStorage.setItem('mca_history', JSON.stringify(history));
  }

  function addMessage(cls, text, save=true, isClarify=false) {
    const m = document.createElement('div');
    m.className = 'message ' + cls;
    if (isClarify) m.classList.add('clarify-message');
    
    let formattedText = text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
    m.innerHTML = formattedText;
    
    const time = document.createElement('span');
    time.className = 'timestamp';
    const now = new Date();
    time.textContent = now.getHours().toString().padStart(2, '0') + ':' + now.getMinutes().toString().padStart(2, '0');
    m.appendChild(time);
    
    chat.appendChild(m);
    chat.scrollTop = chat.scrollHeight;
    
    if (save) saveMessageToHistory(cls, text, isClarify);
  }

  let initialHistory = JSON.parse(localStorage.getItem('mca_history') || '[]');
  initialHistory.forEach(h => addMessage(h.cls, h.text, false, h.isClarify));

  function speak(text, lang) {
    if (MUTED) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang === 'hi' ? 'hi-IN' : (lang === 'pa' ? 'pa-IN' : 'en-US');
    u.onstart = () => { isSpeaking = true; document.querySelector('.header').style.borderBottom = "3px solid var(--accent)"; };
    u.onend = () => { isSpeaking = false; document.querySelector('.header').style.borderBottom = "none"; };
    u.onerror = () => { isSpeaking = false; document.querySelector('.header').style.borderBottom = "none"; };
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  }

  function showSuggestions(arr) {
    if (!arr || !arr.length) return;
    const wrap = document.createElement('div');
    wrap.className = 'suggestions';
    arr.forEach(s => {
      const b = document.createElement('button');
      b.className = 'suggestion';
      b.textContent = s;
      b.onclick = () => { messageInput.value = s; sendMessage(); };
      wrap.appendChild(b);
    });
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    return res.json();
  }

  let typingEl = null;
  function showTyping() {
    if (typingEl) return;
    typingEl = document.createElement('div');
    typingEl.className = 'typing-indicator';
    typingEl.innerHTML = '<span></span><span></span><span></span>';
    chat.appendChild(typingEl);
    chat.scrollTop = chat.scrollHeight;
  }
  
  function hideTyping() {
    if (typingEl) { typingEl.remove(); typingEl = null; }
  }

  function addProfileCard(profile) {
    const card = document.createElement('div');
    card.className = 'profile-card';
    card.innerHTML = `
      <div class="profile-header">
        <strong>Your Profile</strong>
        <button onclick="this.parentElement.parentElement.remove()" style="background:none;border:none;cursor:pointer;">✕</button>
      </div>
      <div>Name: ${profile.name}</div>
      <div>Roll: ${profile.roll}</div>
      <div>Email: ${profile.email}</div>
    `;
    chat.appendChild(card);
    chat.scrollTop = chat.scrollHeight;
  }

  uidSubmit.onclick = async () => {
    const uid = uidInput.value.trim();
    if (!uid) return;
    uidMsg.textContent = '';
    const data = await postJSON('/set_uid', { uid });
    if (data.ok) {
      USER_UID = uid;
      localStorage.setItem('mca_uid', USER_UID);
      uidArea.style.display = 'none';
      if (data.profile) addProfileCard(data.profile);
      addMessage('bot', data.response);
      speak(data.response, 'en');
      if (data.suggestions) showSuggestions(data.suggestions);
    } else {
      uidMsg.textContent = data.message || 'UID not found';
    }
  };

  uidSkip.onclick = () => {
    USER_UID = '';
    localStorage.removeItem('mca_uid');
    uidArea.style.display = 'none';
    addMessage('bot', 'Welcome, Guest.');
    speak('Welcome, Guest.', 'en');
  };

  resetBtn.onclick = async () => {
    try { await fetch('/reset', { method: 'POST' }); } catch(e) {}
    localStorage.removeItem('mca_uid');
    localStorage.removeItem('mca_history');
    USER_UID = '';
    uidArea.style.display = 'flex';
    chat.innerHTML = '';
    addMessage('bot', 'Chat reset.');
  };

  muteBtn.onclick = () => setMuted(!MUTED);

  sendBtn.onclick = sendMessage;
  messageInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });

  async function sendMessage() {
    const txt = messageInput.value.trim();
    if (!txt) return;
    addMessage('user', txt);
    messageInput.value = '';
    showTyping();
    try {
      const data = await postJSON('/chat', { message: txt, uid: USER_UID, language: language.value, session_id: SESSION_ID });
      hideTyping();
      addMessage('bot', data.response, true, data.is_clarify);
      speak(data.response, language.value);
      if (data.suggestions) showSuggestions(data.suggestions);
    } catch(err) {
      hideTyping();
      addMessage('bot', "Something went wrong, please try again.");
    }
  }

  voiceBtn.onclick = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { addMessage('bot', 'Speech recognition not supported.'); return; }
    const r = new SR();
    r.lang = language.value === 'hi' ? 'hi-IN' : (language.value === 'pa' ? 'pa-IN' : 'en-US');
    r.onresult = e => { messageInput.value = e.results[0][0].transcript; sendMessage(); };
    r.start();
  };

})();
