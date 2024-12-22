import os
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.question_gen.openai import OpenAIQuestionGenerator
from llama_index.core.schema import NodeWithScore, TextNode

from ..llms import llm
def transform_and_query(query, cfg, query_engine):
    if cfg.query_transform == "subquery_zeroshot":
        return subquery_zeroshot(query, query_engine)
    elif cfg.query_transform == "subquery_fewshot":
        return subquery_fewshot(query, query_engine)
    else:
        return query_engine.query(transform(llm, query, cfg))


def transform(llm, query, cfg):
    # parser.add_argument('--query_transform', type=str, default='none', help='query transform, available: none, hyde_zeroshot, hyde_fewshot, stepback_zeroshot, stepback_fewshot, subquery_zeroshot, subquery_fewshot')
    if cfg.query_transform == "none":
        return query
    elif cfg.query_transform == "hyde_zeroshot":
        return hyde_zeroshot(query)
    elif cfg.query_transform == "hyde_fewshot":
        return hyde_fewshot(query)
    elif cfg.query_transform == "stepback_zeroshot":
        return stepback_zeroshot(query)
    elif cfg.query_transform == "stepback_fewshot":
        return stepback_fewshot(query)
    else:
        raise Exception("Unknown query transform: %s" % cfg.query_transform)


def hyde(query, prompt_template_str):
    """
    Query改写 - HyDE
    """
    HYDE_PROMPT = PromptTemplate(prompt_template_str, prompt_type=PromptType.SUMMARY)
    hyde = HyDEQueryTransform(include_original=True, hyde_prompt=HYDE_PROMPT)
    query_hyde = hyde.run(query)
    return (
            query_hyde.custom_embedding_strs[0] + " " + query_hyde.custom_embedding_strs[1]
    )


def hyde_zeroshot(query):
    """
    Query改写 - HyDE - zeroshot
    """
    HYDE_TMPL_ZEROSHOT = (
        "Please write a passage to answer the question\n"
        "Try to include as many key details as possible.\n"
        "\n"
        "Here is your task:"
        "Question: {context_str}\n"
        "Passage: "
    )

    return hyde(query, HYDE_TMPL_ZEROSHOT)


def hyde_fewshot(query):
    """
    Query改写 - HyDE - fewshot
    """
    HYDE_TMPL_FEWSHOT = (
        "Please write a passage to answer the question\n"
        "Try to include as many key details as possible.\n"
        "\n"
        "Here are a few examples:\n"
        "\n"
        "\n"
        'Passage:"""\n'
        "1.\n"
        "Question: What is Bel?\n"
        "Passage: Bel is a term that has multiple meanings and can be interpreted in various ways depending on the context. In ancient Mesopotamian mythology, Bel was a prominent deity and the god of the heavens and earth. He was considered the supreme god and was often associated with the city of Babylon. Bel was believed to have control over the forces of nature and was worshipped by the Babylonians through elaborate rituals and offerings.\n\n"
        "In addition to its mythological significance, Bel is also a title given to individuals who hold high positions of authority or leadership. For example, in the ancient Near East, the title of Bel was used to refer to the ruler or king of a city-state. This title denoted power, influence, and the ability to govern and make important decisions.\n\n"
        "Furthermore, Bel is also a term used in the field of linguistics to refer to a unit of sound or phoneme. In phonetics, a bel represents a logarithmic unit used to measure the intensity or loudness of sound. This unit is named after Alexander Graham Bell, the inventor of the telephone, who made significant contributions to the study of sound and communication.\n\n"
        "Overall, the term Bel encompasses a range of meanings and can refer to a mythical deity, a title of authority, or a unit of sound measurement. Its significance varies depending on the context in which it is used, highlighting the diverse nature of this term.\n"
        "2.\n"
        "Question: At It Again contains lyrics co-written by the singer and actor from what city?\n"
        "Passage: The song 'At It Again' features lyrics that were co-written by the singer and actor who hail from the vibrant city of Los Angeles. This city, known for its thriving entertainment industry, has been a hub for countless talented individuals who have made their mark in music, film, and television. The singer and actor, both born and raised in Los Angeles, have been deeply influenced by the diverse and creative atmosphere of their hometown. Their collaboration on 'At It Again' showcases their unique perspectives and storytelling abilities, as they draw inspiration from their personal experiences and the rich cultural tapestry of Los Angeles. Through their lyrics, they paint a vivid picture of the city's energy, its dreams, and its challenges, capturing the essence of their beloved hometown in every verse.\n"
        "3.\n"
        "Question: What year was the English actor and film producer born in who also starred in an adaptation with Renee Zellweger?\n"
        "Passage: The English actor and film producer who starred alongside Renee Zellweger in a film adaptation is Hugh Grant. Born on September 9, 1960, Grant has had a prolific career in the film industry, particularly known for his roles in romantic comedies. He is a native of Hammersmith, London, and he attended the New College at the University of Oxford, where he studied English literature.\n\n"
        "Grant's breakthrough role came in 1994 when he starred in 'Four Weddings and a Funeral.' His performance in this film garnered him international recognition and established his reputation as a leading man in romantic comedy films. His charm and unique comedic timing resonated with audiences worldwide, making him a favorite in the genre.\n\n"
        "In 2001, Grant starred in the film adaptation of 'Bridget Jones's Diary,' alongside American actress Renee Zellweger. The film, based on Helen Fielding's popular novel, was a commercial success, grossing over $280 million worldwide. Grant played the character of Daniel Cleaver, a charming yet unreliable publishing executive who becomes involved in a love triangle with Bridget Jones (Zellweger) and Mark Darcy (Colin Firth).\n\n"
        "'Bridget Jones's Diary' was well-received by critics, and it led to two sequels in which Grant reprised his role. His performance in these films is often considered one of his most memorable. Despite the various challenges in his career, Hugh Grant has remained a significant figure in the film industry for over three decades. His enduring appeal and talent have ensured his place in the annals of cinematic history.\n"
        "4.\n"
        "Question: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?\n"
        "Passage: The 2013 Liqui Moly Bathurst 12 Hour race was held at the iconic Mount Panorama Circuit, situated in Bathurst, New South Wales, Australia. This particular motor racing track is known not only for its challenging layout but also for the spectacular views it offers of the surrounding countryside.\n\n"
        "The Mount Panorama Circuit is a unique blend of seemingly contrasting elements. It is comprised of both long and short straights, fast corners, as well as dramatic ascents and descents, which demand a high level of skill and precision from the drivers. The track's total length is approximately 6.213 kilometers, or about 3.861 miles, a distance that has remained consistent over the years.\n\n"
        "The Mount Panorama Circuit is steeped in motor racing history. It was first opened in 1938 and has since been the venue for a variety of prestigious motor racing events. These include not only the Bathurst 12 Hour, but also the Bathurst 1000, the latter of which is considered one of the most significant events in Australian motorsport.\n\n"
        "The 2013 Liqui Moly Bathurst 12 Hour race was a significant event in the racing calendar. Teams from around the world competed in a gruelling endurance race that tested not only the performance of the cars but also the stamina and skill of the drivers. The race, which started in the early morning darkness, ran for twelve hours, with the victors crossing the finish line as the sun was setting, adding to the drama and spectacle of the event.\n\n"
        "The Mount Panorama Circuit's unique combination of high-speed straights and tight corners, along with its significant changes in elevation, make it one of the most challenging and exciting circuits in the world. Its length and layout have remained largely unchanged since its inception, providing a consistent benchmark for drivers and teams across different racing events. It's a track that demands respect from every driver who races on it, and the 2013 Liqui Moly Bathurst 12 Hour was no exception.\n"
        "5.\n"
        "Question: Which event was a surprise military strike by the Imperial Japanese Navy Air Service while Magruder Hill Tuttle was a senior officer in the United States Navy?\n"
        "Passage: The event that was a surprise military strike by the Imperial Japanese Navy Air Service while Magruder Hill Tuttle was a senior officer in the United States Navy was the infamous attack on Pearl Harbor. On the morning of December 7, 1941, the peaceful tranquility of the Hawaiian naval base was shattered as a fleet of Japanese aircraft descended upon the unsuspecting American forces stationed there. Tuttle, who held a senior position within the US Navy, found himself thrust into the chaos and confusion of the surprise attack.\n\n"
        "The Japanese planes unleashed a devastating assault, targeting the American battleships, cruisers, and aircraft on the island. The attack was swift and relentless, catching the US Navy off guard and causing immense destruction and loss of life. Tuttle, along with his fellow officers and sailors, valiantly fought back against the overwhelming enemy forces, but the surprise nature of the attack made it difficult to mount an effective defense.\n\n"
        "The attack on Pearl Harbor marked a turning point in World War II, propelling the United States into the conflict and forever changing the course of history.\n"
        "\n"
        "Here is your task:"
        "Question: {context_str}\n"
        "Passage: "
    )

    return hyde(query, HYDE_TMPL_FEWSHOT)


def stepback(query, prompt_template_str, llm=llm):
    """
    Query扩写 - Stepback
    """
    stepback_query_gen_prompt = PromptTemplate(prompt_template_str)
    return llm.predict(stepback_query_gen_prompt, query=query) + " " + query


def stepback_zeroshot(query, llm=llm):
    """
    Query扩写 - Stepback - zeroshot
    """
    stepback_query_gen_str_zeroshot = """\
    You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer.
    Just give the content of the Stepback Question.

    Here is your task:
    Original Question: {query}
    Stepback Question: 
    """
    return stepback(query, stepback_query_gen_str_zeroshot, llm=llm)


def stepback_fewshot(query, llm=llm):
    """
    Query拆解 - Stepback - fewshot
    """

    stepback_query_gen_str_fewshot = """\
    You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer.
    Just give the content of the Stepback Question.

    Here are a few examples:

    1.
    Original Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
    Stepback Question: Who are some notable figures that have influenced the names of "The Simpsons" characters?

    2.
    Original Question: Which plant's name is derived from the greek for "bee," Melittis or Hovea?
    Stepback Question: What are the origins of the names for the plants Melittis and Hovea?

    3.
    Original Question: What American media franchise is centered on a series of monster films featuring Godzilla and King Kong?
    Stepback Question: What are some notable American media franchises that feature monster films?

    4.
    Original Question: Tar Creed Superfund site causing toxic areas and therefore reducing the population to what number in November 2010?
    Stepback Question: What impact has the Tar Creek Superfund site had on the population of nearby towns, such as Cardin, Oklahoma?

    5.
    Original Question: Who is likely to have represented and advanced Armenian music more, Art Laboe or Konstantin Orbelyan?
    Stepback Question: Who are some notable figures that have significantly contributed to the representation and advancement of Armenian music?
    
    
    Here is your task:
    Original Question: {query}
    Stepback Question: 
    """

    return stepback(query, stepback_query_gen_str_fewshot, llm=llm)


def subquery(query, prompt_template_str, query_engine):
    query_engine_tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="Data Source",
                description="Data source for the query engine",
            ),
        )
    ]

    subquery_engine = CustomSubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=False,
        question_gen=OpenAIQuestionGenerator.from_defaults(
            prompt_template_str=prompt_template_str
        ),
    )

    return subquery_engine.query(query)


def subquery_zeroshot(query, query_engine):
    """
    Query拆解 - Subquery - zeroshot
    """
    ZEROSHOT_OPENAI_SUB_QUESTION_PROMPT_TMPL = """\
    You are a world class state of the art agent.

    You have access to multiple tools, each representing a different data source or API.
    Each of the tools has a name and a description, formatted as a JSON dictionary.
    The keys of the dictionary are the names of the tools and the values are the \
    descriptions.
    Your purpose is to help answer a complex user question by generating a list of sub \
    questions that can be answered by the tools.

    These are the guidelines you consider when completing your task:
    * Be as specific as possible
    * The sub questions should be relevant to the user question
    * The sub questions should be answerable by the tools provided
    * You can generate multiple sub questions for each tool
    * Tools must be specified by their name, not their description
    * You don't need to use a tool if you don't think it's relevant

    Output the list of sub questions by calling the SubQuestionList function.

    ## Tools
    ```json
    {tools_str}
    ```

    ## User Question
    {query_str}
    """
    return subquery(query, ZEROSHOT_OPENAI_SUB_QUESTION_PROMPT_TMPL, query_engine)


def subquery_fewshot(query, query_engine):
    """
    Query拆解 - Subquery - fewshot
    """
    FEWSHOT_OPENAI_SUB_QUESTION_PROMPT_TMPL = """\
    You are a world class state of the art agent.

    You have access to multiple tools, each representing a different data source or API.
    Each of the tools has a name and a description, formatted as a JSON dictionary.
    The keys of the dictionary are the names of the tools and the values are the \
    descriptions.
    Your purpose is to help answer a complex user question by generating a list of sub \
    questions that can be answered by the tools.

    These are the guidelines you consider when completing your task:
    * Be as specific as possible
    * The sub questions should be relevant to the user question
    * The sub questions should be answerable by the tools provided
    * You can generate multiple sub questions for each tool
    * Tools must be specified by their name, not their description
    * You don't need to use a tool if you don't think it's relevant

    Here are a few examples:

    1.
    Question: What type of media does Tar Creek and Before Stonewall have in common?
    Sub questions:
    - What is the media type of Tar Creek?
    - What is the media type of Before Stonewall?

    2.
    Question: Which documentary was created first, Murderball or Year at Danger?
    Sub questions: 
    - When was the documentary "Murderball" created?
    - When was the documentary "Year at Danger" created?

    3.
    Question: How was Paul Grahams life different before, during, and after YC?
    Sub questions:
    - What did Paul Graham work on before YC?
    - What did Paul Graham work on during YC?
    - What did Paul Graham work on after YC?

    4.
    Question: What role did Ed Roland and Adam Duritz perform in their respective bands, Collective Soul and Counting Crows?
    Sub questions: 
    - What role did Ed Roland perform in the band Collective Soul?
    - What role did Adam Duritz perform in the band Counting Crows?

    5.
    Question: Did Down and Out in America or Lost in La Mancha win more awards?
    Sub questions: 
    - How many awards did the movie "Down and Out in America" win?
    - How many awards did the movie "Lost in La Mancha" win?

    Output the list of sub questions by calling the SubQuestionList function.

    ## Tools
    ```json
    {tools_str}
    ```

    ## User Question
    {query_str}
    """
    return subquery(query, FEWSHOT_OPENAI_SUB_QUESTION_PROMPT_TMPL, query_engine)


class CustomSubQuestionQueryEngine(SubQuestionQueryEngine):
    def _construct_node(self, qa_pair):
        node_text = f"Sub question: {qa_pair.sub_q.sub_question}\nResponse: {qa_pair.answer}"
        if qa_pair.sources and len(qa_pair.sources) > 0:
            metadata = qa_pair.sources[0].node.metadata.copy()
        else:
            metadata = {}
        node = TextNode(text=node_text, metadata=metadata)
        return NodeWithScore(node=node)
