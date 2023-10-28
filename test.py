import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from llm_few_shot_gen.generators.midjourney import MidjourneyPromptGenerator
from llm_few_shot_gen.few_shot_examples.utils import get_shirt_design_prompt_examples

llm = ChatOpenAI(temperature=0.7)
prompt_generator = MidjourneyPromptGenerator(llm)
prompt_generator._set_context()
prompt_examples = get_shirt_design_prompt_examples()
prompt_generator.set_few_shot_examples(prompt_examples)

text = """Pew Madafakas Tshirt is a Funny Black Cat with Guns Tee for Lovers of Sarcasm, Joke, Irony - men, women, dad, mom, friend, wife, husband. 
Great Birthday Gift for cat lover, cat owner, cat mom, cat dad, cat mama, cat papa for casual or special wearing Pew Madafakas T-Shirt great pairs with funny cat lover accessories, stuff, supplies, clothes, decorations, long sleeve, cap, hat, collar, sticker, decal, shorts, mug, cup, balloons, bandana.
 Great cat lover gifts for women funny cat owner must haves"""
text2 = "Drücken Sie sich mit dieser lustigen, motivierenden und neuartigen Bekleidung aus. Wenn Sie nach perfekten Damenhemden, Herrenhemden, lustigen Shirts mit einem weichen Gefühl suchen, die Sie den ganzen Tag tragen können, dann ist dieses T-Shirt genau das Richtige für Sie Perfekt für Geburtstage, Vatertag für Vater, Muttertag für Mutter, 4. Juli, Jahrestag oder jede andere Gelegenheit. Dies ist das beste Geschenk für Sie, Ihren Freund, Bruder, Schwester, Großeltern, Klassenkameraden, Büromitarbeiter, Lehrer usw. Cat Vintage PewPewPew Madafakas Cat Crazy Pew Vintage Funny Cat Lovers Macht ein großes Geschenk für Sie und Ihn. Beste Geschenkidee für Freund & Freundin, Ehemann & Ehefrau, Mama & Papa zum Geburtstag, Jahrestag, Valentinstag, Weihnachten. Cat Vintage PewPewPew Madafakas tee, Cat Crazy Pew Vintage tee, eine einfache Frau, die die Schwarze Katze liebt Liebesgeschenk, süße Schwarze Katze Großes Geschenk für Vater, Vater, Mutter, und die Katzen liebt"
prompt_generator.generate(text=text)

human_message_template = """
I want you to act as a professional image ai user. 
Write a single concise english prompt for the text delimited by ```. 
The output prompt should focus on visual descriptions. 
Take inspiration from the formating from the example prompts, dont copy them, but use the same format.
Your output should only contain the single prompt without further details.
```{text}```
"""
prompt_generator.messages.io_prompt = HumanMessagePromptTemplate.from_template(human_message_template)
prompt_generator._get_llm_chain().run(text=text2)