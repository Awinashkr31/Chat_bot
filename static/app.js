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

  function setMuted(v) {
    MUTED = !!v;
    localStorage.setItem('mca_muted', MUTED ? '1' : '0');
    muteBtn.textContent = MUTED ? 'Unmute' : 'Mute';
    if (MUTED && window.speechSynthesis) window.speechSynthesis.cancel();
  }
  setMuted(MUTED);

  function addMessage(cls, text) {
    const m = document.createElement('div');
    m.className = 'message ' + cls;
    m.textContent = text;
    chat.appendChild(m);
    chat.scrollTop = chat.scrollHeight;
  }

  function speak(text, lang) {
    if (MUTED) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang === 'hi' ? 'hi-IN' : (lang === 'pa' ? 'pa-IN' : 'en-US');
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

  uidSubmit.onclick = async () => {
    const uid = uidInput.value.trim();
    if (!uid) return;
    uidMsg.textContent = '';
    const data = await postJSON('/set_uid', { uid });
    if (data.ok) {
      USER_UID = uid;
      localStorage.setItem('mca_uid', USER_UID);
      uidArea.style.display = 'none';
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
    await fetch('/reset', { method: 'POST' });
    localStorage.removeItem('mca_uid');
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
    const data = await postJSON('/chat', { message: txt, uid: USER_UID, language: language.value });
    addMessage('bot', data.response);
    speak(data.response, language.value);
    if (data.suggestions) showSuggestions(data.suggestions);
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
