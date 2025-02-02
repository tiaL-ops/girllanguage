import { useState } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    { text: "Hey girl! 💖 How's your day?", sender: "bot" },
  ]);
  const [input, setInput] = useState("");

  const sendMessage = () => {
    if (input.trim() !== "") {
      setMessages([...messages, { text: input, sender: "user" }]);
      setInput("");

      // Simulating a cute response
      setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          { text: "That sounds amazing! 🌸 Tell me more!", sender: "bot" },
        ]);
      }, 1000);
    }
  };

  return (
    <div className="chat-container">
      <h1>💬 Girl Talk Chat 💖</h1>
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            {msg.text}
          </div>
        ))}
      </div>
      <div className="input-box">
        <input
          type="text"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button onClick={sendMessage}>Send 💌</button>
      </div>
    </div>
  );
}

export default App;
