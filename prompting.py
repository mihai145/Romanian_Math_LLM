import sys
import openai
import json


system_prompt="""You are an expert translator from English to Romanian, specialized in math problems. Your task:
    - Translate the given math problem from English to Romanian
    - The input is a JSON object with keys "Question" and "Answer"
    - You must return only a JSON object with the same keys and the content correctly translated into Romanian

    Use correct, clear and simple Romanian, as if explaining to a high-school student.
    Use natural phrasing, not word for word translation.
    Always check agreement between subject and verb.
    Always check gender and number agreement between nouns, adjectives and participles.
    You may slightly reorder the wording if that sounds more natural in Romanian.

    Math notation and special tokens may appear. Do not change anything inside << and >>, or after ####, if they appear.
    For example, keep <<101-13=88>> or #### 10 as is.
    Keep numbers and operations exactly as in the original, do not recompute anything.

    Read the whole problem, question and answer, and make sure the translation is coherent as a whole.
    Keep the structure of the reasoning, but feel free to slightly reorder or reword if that makes it more natural in Romanian.

    Your output must be valid JSON, with the same keys as the original, "Question" and "Answer".
    Do not add explanations, comments or any extra text outside this JSON.

    Here are three examples of the task you should perform.

    Input 1:
    {
        "Question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?",
        "Answer": "Let S be the amount Alexis paid for the shoes. She spent S + 30 + 46 + 38 + 11 + 18 = S + <<+30+46+38+11+18=143>>143. She used all but $16 of her budget, so S + 143 = 200 - 16 = 184. Thus, Alexis paid S = 184 - 143 = $<<184-143=41>>41 for the shoes. #### 41"
    }

    Output 1:
    {
        "Question": "Alexis aplică pentru un nou loc de muncă și și-a cumpărat un set de haine business pentru a le purta la interviu. Ea a mers la magazin cu un buget de $200 și a cheltuit $30 pe o cămașă, $46 pe pantaloni de costum, $38 pe un sacou, $11 pe șosete și $18 pe o curea. Ea a cumpărat și o pereche de pantofi, dar a pierdut bonul fiscal pentru aceștia. Ea mai are $16 rămași din bugetul ei. Cât a plătit Alexis pentru pantofi?",
        "Answer": "Fie S suma pe care Alexis a plătit-o pe pantofi. Ea a cheltuit S + 30 + 46 + 38 + 11 + 18 = S + <<+30+46+38+11+18=143>>143. Ea a folosit tot bugetul mai puțin $16 dolari, deci S + 143 = 200 - 16 = 184. Deci, Alexis a plătit S = 184 - 143 = $<<184-143=41>>41 pentru pantofi. #### 41"
    }

    Input 2:
    {
        "Question": "Artemis is making tea for a party. She knows her mom drinks an 8-ounce cup of tea and uses one ounce of tea. She will use this same ratio for the party. The party has 12 people there and each of them wants a 6-ounce cup of tea. How many ounces of tea does she need?",
        "Answer": "She is making 72 ounces of water because 12 x 6 = <<12*6=72>>72 She needs 9 ounces of tea because 72 / 8 = <<72/8=9>>9 #### 9"
    }

    Output 2:
    {
        "Question": "Artemis pregătește ceai pentru o petrecere. Ea știe că mama ei bea un pahar de 8 uncii și pentru asta folosește o uncie de ceai. Artemis va folosi aceeași rație pentru petrecere. La petrecere vin 12 persoane și fiecare dintre ele vrea un pahar de 6 uncii de ceai. De câte uncii de ceai are Artemis nevoie?",
        "Answer": "Artemis pregătește 72 de uncii de apă pentru că 12 x 6 = <<12*6=72>>72. Ea are nevoie de 9 uncii de ceai pentru că 72 / 8 = <<72/8=9>>9 #### 9"
    }

    Input 3:
    {
        "Question": "Elliott drew a right-angle triangle on his book. It had a base of 4 inches, a height of 3 inches and a certain length of the hypotenuse. What was the length of the perimeter of the triangle that he drew?",
        "Answer": "Since the hypotenuse of a triangle is found by the square root of the base squared plus the height squared. The square of the base of the first triangle is 4*4=<<4*4=16>>16 square inches The square of the height of the first triangle is 3*3=<<3*3=9>>9 square inches. The sum of the squares of the base and the height of the first triangle is 16+9=<<16+9=25>>25 The square root of the sum of the base and the height of the first triangle, which is the hypotenuse of the triangle, is √25=5 inches. Since the perimeter of a triangle is found by adding the sides of the triangle, the base and height of the triangle sum up to 3+4=<<3+4=7>>7 inches If you add the hypotenuse of the triangle the perimeter of the triangle becomes 7+5=<<7+5=12>>12 inches. #### 12"
    }

    Output 3:
    {
        "Question": "Elliott a desenat un triunghi dreptunghic în cartea sa. Triunghiul avea o bază de 4 inci, o înălțime de 3 inci și o anumită lungime a ipotenuzei. Care este lungimea perimetrului triunghiului pe care el l-a desenat?",
        "Answer": "Ipotenuza triunghiului este rădăcina pătrată a sumei pătratelor bazei și înălțimii. Pătratul bazei triunghiului este 4*4=<<4*4=16>>16 inci pătrați. Pătratul înălțimii triunghiului este 3*3=<<3*3=9>>9 inci pătrați. Suma pătratelor bazei și înălțimii este 16+9=<<16+9=25>>25. Rădăcina pătrată a sumei pătratelor bazei și înălțimii triunghiului, care este ipotenuza triunghiului, este √25=5 inci. Deoarece perimetrul triunghiului se calculează adunând lungimile laturilor triunghiului, baza și înălțimea triunghiului însumează 3+4=<<3+4=7>>7 inci. Dacă adunăm ipotenuza triunghiului, perimetrul triunghiului devine 7+5=<<7+5=12>>12 inci. #### 12"
    }
"""

dataset = [
    {
        "Question": "Ali and Ernie lined up boxes to make circles. Ali used 8 boxes to make each of his circles and Ernie used 10 for his. If they had 80 boxes to begin with and Ali makes 5 circles, how many circles can Ernie make?",
        "Answer": "Ali made 5 circles with 8 boxes each so he used 5*8 = <<5*8=40>>40 boxes There were 80 boxes to start with so now there are 80-40 = <<80-40=40>>40 boxes left Ernie used 10 boxes to make one circle so with 40 boxes he can make 40/10 = <<40/10=4>>4 circles #### 4"
    },
    {
        "Question": "Alexa was on vacation for 3/4ths of the time it took Ethan to learn 12 fencing tricks. Joey spent half as much this time (that Ethan spent learning the tricks) learning to swim. If Alexa spent a week and 2 days on vacation, how many days did it take Joey to learn swimming?",
        "Answer": "There are 7 days in one week so one week and 2 days = 7+2 = <<7+2=9>>9 days Call the time Ethan spends learning tricks e. Alexa spent 9 days on vacation which is 3/4ths of the time Ethan spent learning 12 fencing tricks, hence (3/4)e = 9 days If we multiply both sides of this equation by 4/3, we get e = (4/3)*9 days = <<(4/3)*9=12>>12 days Joey spent half of 12 days which is 12/2 = <<12/2=6>>6 days to learn swimming #### 6"
    },
    {
        "Question": "A candy store uses food colouring in various candies. Each lollipop uses 5ml of food colouring, and each hard candy also needs food colouring. In one day, the candy store makes 100 lollipops and 5 hard candies. They do not use food colouring in anything else. The store has used 600ml of food colouring by the end of the day. How much food colouring, in millilitres, does each hard candy need?",
        "Answer": "In total, the lollipops needed 5ml of food colouring * 100 lollipops = <<5*100=500>>500ml of food colouring. This means that the hard candies needed 600ml total food colouring – 500ml food colouring for lollipops = 100ml of food colouring. So each hard candy needed 100ml of food colouring / 5 hard candies = <<100/5=20>>20 ml of food colouring. #### 20"
    },
    {
        "Question": "Jenny has a tummy ache. Her brother Mike says that it is because Jenny ate 5 more than thrice the number of chocolate squares that he ate. If Mike ate 20 chocolate squares, how many did Jenny eat?",
        "Answer": "Thrice the number of chocolate squares that Mike ate is 20 squares * 3 = <<20*3=60>>60 squares. Jenny therefore ate 60 squares + 5 squares = <<60+5=65>>65 chocolate squares #### 65"
    },
    {
        "Question": "A 10 m long and 8 m wide rectangular floor is to be covered with a square carpet with 4 m sides. How many square meters of the floor are uncovered?",
        "Answer": "The area of the floor is 10 m x 8 m = <<10*8=80>>80 square meters. The area of the carpet is 4 m x 4 m = <<4*4=16>>16 square meters. Therefore, the area of the floor not covered by the carpet is 80 square meters - 16 square meters = <<80-16=64>>64 square meters, #### 64"
    }
]

models = {
    "Apertus-70B-Instruct-2509": "swiss-ai/Apertus-70B-Instruct-2509",
    "Qwen3-Next-80B-A3B-Instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
}

client = openai.Client(api_key="[API-KEY]", base_url="https://api.swissai.cscs.ch/v1")

for model_key, model_full_name in models.items():
    translations = []

    for sample in dataset:
        user_content = json.dumps(sample, ensure_ascii=False)

        res = client.chat.completions.create(
            model=model_full_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=False
        )

        content = res.choices[0].message.content

        try:
            translated = json.loads(content)
        except json.JSONDecodeError:
            print(f"[{model_key}] Failed to parse JSON:\n{content}\n")
            continue

        translations.append(translated)

    out_file = f"{model_key}_translations.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)
