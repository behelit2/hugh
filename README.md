Hugh: The Command Line Chatbot. 

Hugh is a command line chatbot that's been built on the Microsoft Phi-2 Research Model. It's designed to assist you with any queries or tasks you propose, providing a seamless blend of question-answering, instruction-following, and creative command execution capabilities in a single conversational interface.

1. Downloading the Model. You need to get the Phi-2 Model to use with Hugh. Go to the Hugging Face page for Phi-2. The URL is https://huggingface.co/microsoft/phi-2 You will need to download the model to a location on your local pc or server. Instructions for that here https://huggingface.co/docs/hub/en/models-downloading

2. Referencing the Model and running hugh01.py. Once you have successfully downloaded the model, you must point to the path of this model in hugh01.py. Locate the [model_path = "./phi-2"] row and ensure it is pointed to the correct location. Once your path is correct, simply run hugh01.py at the command line like so:

   python3 hugh01.py

3. Purpose. The primary goal for Hugh is to serve as a foundation for more advanced chatbots, aiming to undertake valuable operational tasks and coding at the OS level. It excels in various areas and provides satisfactory performance on most benchmarks.

4. Using Hugh. Interacting with Hugh primarily involves posing queries, providing instructions or setting creative tasks. More intricate concepts might require multiple prompting for accurate comprehension. However, Hugh learns and adapts quickly, specific examples significantly aid its understanding. Hugh has basic datetime support and should be able to accurately reference the date or time using the system clock in conversations. Further, all conversations are saved in chatlog.txt for easy reference and review. Typing /clear allows you to clear the conversation buffer, and /quit allows you to gracefully exit the application.

5. Acknowledgement. Your feedback and contributions to this project are highly appreciated. Together we can help Hugh grow and perform far better.

6. License. This project is MIT license, further details can be found in the LICENSE file included in this repository.
