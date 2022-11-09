
function get(selector, root = document) {
    return root.querySelector(selector);
  }
  
 
const msgerForm = get('.msger-inputarea');
const msgerInput = get('.msger-input');
const msgerChat = get('.msger-chat');

const BOT_IMG = "https://media.istockphoto.com/vectors/chat-bot-ai-and-customer-service-support-concept-vector-flat-person-vector-id1221348467?k=20&m=1221348467&s=170667a&w=0&h=7gndncDuLvMy_RmSEc-NyWmgA3x3KA96Aj5q5Xx-wTE=";
const PERSON_IMG =
  "https://lh3.googleusercontent.com/5t4_DYooTUgRMG4x7KrRXia_okdGSO5Cm7OCg53AgcrKbOa_SOfQ2fSFDmHOzdcZL8ZqJKx_1_hXOjqWumBGFhW_=s2007";
const BOT_NAME = "BOT";
const PERSON_NAME = "Satyam";

 
console.log("meessage form ",msgerForm)
console.log("message input", msgerInput)
console.log("message cahat " , msgerChat)
const BOT_MSGS = [
  "Hi, how are you? ",
  "Ohh... I can't understand what you trying to say. Sorry!",
  "I like to play games... But I don't know how to play!",
  "Sorry if my answers are not relevant. :))",
  "I feel sleepy! :(",
];
 

msgerForm.addEventListener('submit', (event) => {
    event.preventDefault();
  console.log(event)
    const msgText = msgerInput.value;
    if (!msgText) return;
  
    appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);
    msgerInput.value = "";
  
    botResponse();
  });
  


function appendMessage(name, img, side, text) {
     
    const msgHTML = `
      <div class="msg ${side}-msg">
        <div class="msg-img" style="background-image: url(${img})"></div>
  
        <div class="msg-bubble">
          <div class="msg-info">
            <div class="msg-info-name">${name}</div>
            <div class="msg-info-time">${formatDate(new Date())}</div>
          </div>
  
          <div class="msg-text">${text}</div>
        </div>
      </div>
    `;
  
    msgerChat.insertAdjacentHTML("beforeend", msgHTML);
    msgerChat.scrollTop += 500;
  }

function botResponse() {
  const r = random(0, BOT_MSGS.length - 1);
  const msgText = BOT_MSGS[r];
  const delay = msgText.split(" ").length * 100;

  setTimeout(() => {
    appendMessage(BOT_NAME, BOT_IMG, "left", msgText);
  }, delay);
}



function formatDate(date) {
  const h = "0" + date.getHours();
  const m = "0" + date.getMinutes();

  return `${h.slice(-2)}:${m.slice(-2)}`;
}

function random(min, max) {
  return Math.floor(Math.random() * (max - min) + min);
}
