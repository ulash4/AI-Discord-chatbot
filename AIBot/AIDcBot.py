import discord
import os
from dotenv import load_dotenv
from discord.ext import commands
import random
import requests
import json
import torch
from model import NeuralNetwork
from nltk_utils import BagOfWords, Tokenize

load_dotenv('DC_TOKEN.env')
TOKEN = os.getenv("DISCORD_TOKEN")

#client = discord.Client()
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    guild_count = 0
    for guild in bot.guilds:
        print(f"- {guild.id} (name: {guild.name})")
        guild_count += 1

@bot.event
async def on_message(message):
    if message.author == bot.user:  # Ignore messages from the bot itself
        return

    content = message.content.strip().lower()
    
    if content == "naber":
        print("Received 'sa'")
        await message.channel.send("iyiyim senden " + message.author.mention)
    

    await bot.process_commands(message)

@bot.command()
async def hello(ctx):
    await ctx.send(f"Hello {ctx.author.mention}!")

 # bot'un bir siteden html parse yaparak gif göndermesini sağlıyor   
@bot.command()
async def gif(ctx,*arg):
    gif_endpoint = f"https://tenor.com/tr/search/{arg}-gifs"
    req = requests.get(gif_endpoint)

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(req.text,"html.parser")
    
    results = soup.find_all("div", attrs={"class":"Gif"})
    #print(results)
    img_source = []
    for img_tag in results:
        images = img_tag.find_all("img")
        for img in images:
            src = img.get("src")
            img_source.append(src)
            print(src)    
    
    await ctx.send(random.choice(img_source))

#İşte sohbet edebileceğiniz AI modelimiz
#chatbot ile konuşmak için !chat blabla yazarak sohbete başlıyoruz
@bot.command()
async def chat(ctx,*,args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("intents.json","r") as f:
        intents = json.load(f)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNetwork(input_size,hidden_size,output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    sentence = Tokenize(args)
    print(sentence)
    X = BagOfWords(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim = 1)
    probs = probs[0][predicted.item()]

    if probs.item() > 0.60:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                await ctx.send(random.choice(intent["responses"]))
                #print(random.choice(intent["responses"]))
    else:
        print("I don't understand that dude")


bot.run(TOKEN)

