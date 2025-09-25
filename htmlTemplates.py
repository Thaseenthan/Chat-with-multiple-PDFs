import base64

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Convert images into base64 strings
bot_img = img_to_base64("docs/AI.jpg")
user_img = img_to_base64("docs/Human.webp")

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/jpg;base64,{bot_img}" alt="AI Avatar">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
'''

user_template = f'''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/webp;base64,{user_img}" alt="User Avatar">
    </div>    
    <div class="message">{{{{MSG}}}}</div>
</div>
'''
