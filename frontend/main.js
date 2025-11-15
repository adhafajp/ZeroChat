const chatWindow = document.getElementById('chat-window')
const chatForm = document.getElementById('chat-form')
const promptInput = document.getElementById('prompt-input')
const sendButton = document.getElementById('send-button')

const STREAM_URL = '/api/stream-sse' 

let characterQueue = []
let isTyping = false
let typingInterval = null
let currentBotResponse = ''
let currentTextNode = null

function startTypingAnimation () {
  if (isTyping) return
  isTyping = true

  typingInterval = setInterval(() => {
    if (characterQueue.length > 0) {
      const char = characterQueue.shift()
      currentBotResponse += char
      if (currentTextNode) {
        currentTextNode.textContent += char
      }
      
      // Auto-scroll
      const isUserNearBottom =
        chatWindow.scrollHeight -
          chatWindow.scrollTop -
          chatWindow.clientHeight <
        100
      if (isUserNearBottom) {
        chatWindow.scrollTop = chatWindow.scrollHeight
      }

    } else {
      clearInterval(typingInterval)
      isTyping = false
    }
  }, 10)
}

chatForm.addEventListener('submit', async e => {
  e.preventDefault()

  const prompt = promptInput.value.trim()
  if (!prompt || isTyping) return

  promptInput.value = ''
  promptInput.disabled = true
  sendButton.disabled = true

  addMessageToChat('user', prompt)
  const botBubble = addMessageToChat('bot')
  const contentDiv = botBubble.querySelector('.markdown-content')

  const textWrapper = document.createElement('p')
  textWrapper.className = 'typing-text-wrapper'
  currentTextNode = document.createElement('span')
  const cursorSpan = document.createElement('span')
  cursorSpan.className = 'blinking-cursor'
  textWrapper.appendChild(currentTextNode)
  textWrapper.appendChild(cursorSpan)
  contentDiv.innerHTML = ''
  contentDiv.appendChild(textWrapper)
  chatWindow.scrollTop = chatWindow.scrollHeight

  characterQueue = []
  currentBotResponse = ''
  if (typingInterval) clearInterval(typingInterval)
  isTyping = false

  try {
    const response = await fetch(STREAM_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify({ text: prompt })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let partialLine = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }

      const chunk = decoder.decode(value, { stream: true })
      
      const lines = (partialLine + chunk).split('\n\n')
      partialLine = lines.pop()

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const sseData = line.substring(6).trim()
          if (!sseData) continue

          const data = JSON.parse(sseData)
          const token = data.token

          if (token === '[DONE]') {
            await waitForTypingToFinish(cursorSpan, contentDiv, botBubble)
            return
          
          } else if (token.startsWith('[ERROR]')) {
            throw new Error(token)
          
          } else {
            characterQueue.push(...token.split(''))
            startTypingAnimation()
          }
        }
      }
    }
    
    await waitForTypingToFinish(cursorSpan, contentDiv, botBubble)

  } catch (error) {
    console.error('Error while streaming:', error)
    contentDiv.innerHTML = `<p>Sorry, an error occurred: ${error.message}</p>`
    botBubble.classList.remove('bg-blue-600')
    botBubble.classList.add('bg-red-500')
    
    if (typingInterval) clearInterval(typingInterval)
    isTyping = false
    characterQueue = []
    cursorSpan.remove()
    promptInput.disabled = false
    sendButton.disabled = false
    promptInput.focus()
  }
})

function waitForTypingToFinish (cursorSpan, contentDiv, botBubble) {
  return new Promise(resolve => {
    const checkInterval = setInterval(() => {
      if (!isTyping) {
        clearInterval(checkInterval)
        
        cursorSpan.remove()

        const finalHtml = marked.parse(currentBotResponse)
        contentDiv.innerHTML = DOMPurify.sanitize(finalHtml, {
          USE_PROFILES: { html: true }
        })

        botBubble.querySelectorAll('pre code').forEach(block => {
          hljs.highlightElement(block)
        })

        promptInput.disabled = false
        sendButton.disabled = false
        promptInput.focus()
        
        resolve()
      }
    }, 50)
  })
}

function addMessageToChat (sender, text = '') {
  const messageWrapper = document.createElement('div')
  messageWrapper.classList.add('chat-message', sender)
  let bgClass = ''
  let roundedClass = ''
  let contentHtml = ''
  if (sender === 'user') {
    bgClass = 'bg-gray-200'
    roundedClass = 'rounded-br-none'
    messageWrapper.classList.add('flex', 'justify-end')
    const p = document.createElement('p')
    p.innerText = text
    contentHtml = p.outerHTML
  } else {
    bgClass = 'bg-blue-600 text-white'
    roundedClass = 'rounded-bl-none'
    messageWrapper.classList.add('flex', 'justify-start')
    contentHtml = '<div class="markdown-content text-white max-w-none"></div>'
  }
  const bubble = document.createElement('div')
  bubble.className = `message-bubble p-4 ${bgClass} rounded-2xl ${roundedClass} max-w-xl shadow-lg`
  bubble.innerHTML = contentHtml
  messageWrapper.appendChild(bubble)
  chatWindow.appendChild(messageWrapper)

  chatWindow.scrollTop = chatWindow.scrollHeight
  return bubble
}

promptInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    chatForm.requestSubmit()
  }
})