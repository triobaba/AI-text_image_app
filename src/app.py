import os
import tkinter as tk
import customtkinter as ctk
from customtkinter import CTkImage
from auth_token import AuthToken
from PIL import Image,ImageTk
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
import torch
from torch import  autocast







app=tk.Tk()
app.geometry("532x644")
app.title("Text to Image generator")
ctk.set_appearance_mode=("dark")

text_input=ctk.CTkEntry(height=40,width=512,text_color="black",fg_color="white",master=app)
text_input.place(x=10,y=18)

#lmain=ctk.CTkLabel(height=512, width=512, master=app)
#lmain.place(x=9, y=70)



def generate_image():
    model_path = "/Users/triobaba/model.pt_1"
    input_text = text_input.get()

    #i
    
       
 # Download the model and save it to a file
   


    if os.path.isfile(model_path):
        #Load the model from the file
        model = torch.load(model_path)
    else:
        model_id="stabilityai/stable-diffusion-2" #downloads the model from the huggingface model hub
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model= DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", scheduler=scheduler,torch_dtype=torch.float32, use_safetensors=True, variant="fp16",use_auth_token=AuthToken)
        torch.save(model, model_path)
        num_images=2

        images_per_row=1

    frame = tk.Frame(app)
    frame.place(x=10, y=90)

    canvas=tk.Canvas(frame, width=512, height=512)
    scrollbar=tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left")
    #canvas.place(x=10, y=90)

    for i in range(num_images):

        with torch.no_grad():
            image_output = model(input_text).images[0]
    # Display the generated image
         # Convert the image data to a PIL Image
            image_pil = image_output
            image_pil=image_pil.resize((512,512), Image.LANCZOS)
       
        
        
        #new_image= (image_pil.width, image_pil.height)
        #image_pil=image_pil.resize(new_image)
        #image_ctk=CTkImage(image_pil)
        image_tk=ImageTk.PhotoImage(image_pil)

        #lmain.configure(image=image_ctk)
        #lmain.image = image_ctk
        #label=tk.Label(app, image=image_tk)
        #label.pack()

        x = (i % images_per_row) * 512
        y = (i // images_per_row) * 512

  
        #app.config(canvas=canvas)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        canvas.image = image_tk

        app.update_idletasks()
        output_file_path="/Users/triobaba/generatedimages.png"
        if os.path.isfile(output_file_path):
            image_pil.save="generatedimages.png"
        else:
            print(f"cannot save image: No write access to {os.path.dirname(output_file_path)}")
    canvas.config(scrollregion=canvas.bbox("all"))

button=ctk.CTkButton(master=app, text="Generate Image", command=generate_image)
button.place(x=10, y=60)
       
app.mainloop()