import datasets
from lm_eval.tasks.sciq import SciQ

from iz.crossfit import kilt_hotpotqa, kilt_nq
from iz.crossfit.acronym_identification import AcronymIdentification
from iz.crossfit.ade_classification import AdeCorpusV2_Classfication
from iz.crossfit.ade_dosage import AdeCorpusV2_Dosage
from iz.crossfit.ade_effect import AdeCorpusV2_Effect
from iz.crossfit.adversarial_qa import AdversarialQA
from iz.crossfit.aeslc import AESLC
from iz.crossfit.agnews import AGNews
from iz.crossfit.ai2_arc import AI2_ARC
from iz.crossfit.amazon_polarity import AmazonPolarity
from iz.crossfit.anli import ANLI
from iz.crossfit.app_reviews import AppReviews
from iz.crossfit.aqua_rat import AquaRat
from iz.crossfit.art import ART
from iz.crossfit.aslg_pc12 import ASLG_PC12
from iz.crossfit.biomrc import BioMRC
from iz.crossfit.blimp import BLIMP
from iz.crossfit.boolq import BoolQ
from iz.crossfit.break_task import Break
from iz.crossfit.circa import Circa
from iz.crossfit.climate_fever import ClimateFever
from iz.crossfit.codah import CODAH
from iz.crossfit.commongen import CommonGen
from iz.crossfit.commonsense_qa import CommonsenseQA
from iz.crossfit.cos_e import CoS_E
from iz.crossfit.cosmos_qa import CosmosQA
from iz.crossfit.crawl_domain import CrawlDomain
from iz.crossfit.crows_pairs import CrowsPairs
from iz.crossfit.dbpedia_14 import DBpedia14
from iz.crossfit.definite_pronoun_resolution import DefinitePronounResolution
from iz.crossfit.discovery import Discovery
from iz.crossfit.dream import Dream
from iz.crossfit.duorc import DuoRC
from iz.crossfit.e2e_nlg_cleaned import E2E_NLG
from iz.crossfit.eli5 import ELI5
from iz.crossfit.emo import Emo
from iz.crossfit.emotion import Emotion
from iz.crossfit.empathetic_dialogues import EmpatheticDialogues
from iz.crossfit.ethos import Ethos
from iz.crossfit.fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset, \
    FewshotGymTextToTextDataset
from iz.crossfit.financial_phrasebank import FinancialPhrasebank
from iz.crossfit.freebase_qa import FreebaseQA
from iz.crossfit.gigaword import Gigaword
from iz.crossfit.glue_cola import Glue_Cola
from iz.crossfit.glue_mnli import Glue_MNLI
from iz.crossfit.glue_mrpc import Glue_MRPC
from iz.crossfit.glue_qnli import Glue_QNLI
from iz.crossfit.glue_qqp import Glue_QQP
from iz.crossfit.glue_rte import Glue_RTE
from iz.crossfit.glue_sst2 import Glue_SST2
from iz.crossfit.glue_wnli import Glue_WNLI
from iz.crossfit.google_wellformed_query import GoogleWellformedQuery
from iz.crossfit.hate_speech18 import HateSpeech18
from iz.crossfit.hate_speech_offensive import HateSpeechOffensive
from iz.crossfit.hatexplain import HatExplain
from iz.crossfit.health_fact import HealthFact
from iz.crossfit.hellaswag import HellaSwag
from iz.crossfit.hotpot_qa import HotpotQA
from iz.crossfit.imdb import IMDB
from iz.crossfit.jeopardy import Jeopardy
from iz.crossfit.kilt_ay2 import Kilt_AY2
from iz.crossfit.kilt_fever import Kilt_Fever
from iz.crossfit.kilt_trex import Kilt_TREX
from iz.crossfit.kilt_wow import Kilt_WoW
from iz.crossfit.kilt_zsre import Kilt_ZSRE
from iz.crossfit.lama import LAMA
from iz.crossfit.liar import Liar
from iz.crossfit.limit import Limit
from iz.crossfit.math_qa import MathQA
from iz.crossfit.mc_taco import MC_TACO
from iz.crossfit.medical_questions_pairs import MedicalQuestionPairs
from iz.crossfit.mocha import Mocha
from iz.crossfit.multi_news import MultiNews
from iz.crossfit.numer_sense import NumerSense
from iz.crossfit.onestop_english import OneStopEnglish
from iz.crossfit.openbookqa import OpenbookQA
from iz.crossfit.paws import PAWS
from iz.crossfit.piqa import PIQA
from iz.crossfit.poem_sentiment import PoemSentiment
from iz.crossfit.proto_qa import ProtoQA
from iz.crossfit.qa_srl import QA_SRL
from iz.crossfit.qasc import QASC
from iz.crossfit.quail import QUAIL
from iz.crossfit.quarel import QUAREL
from iz.crossfit.quartz import Quartz
from iz.crossfit.quoref import Quoref
from iz.crossfit.race import Race
from iz.crossfit.reddit_tifu import Reddit_TIFU
from iz.crossfit.ropes import ROPES
from iz.crossfit.rotten_tomatoes import RottenTomatos
from iz.crossfit.samsum import SAMSum
from iz.crossfit.scicite import SciCite
from iz.crossfit.scitail import SciTail
from iz.crossfit.search_qa import SearchQA
from iz.crossfit.sick import Sick
from iz.crossfit.sms_spam import SMS_Spam
from iz.crossfit.social_i_qa import SocialIQA
from iz.crossfit.spider import Spider
from iz.crossfit.squad import SQuAD
from iz.crossfit.superglue_cb import Superglue_CB
from iz.crossfit.superglue_copa import Superglue_COPA
from iz.crossfit.superglue_multirc import Superglue_MultiRC
from iz.crossfit.superglue_record import Superglue_Record
from iz.crossfit.superglue_rte import Superglue_RTE
from iz.crossfit.superglue_wic import Superglue_Wic
from iz.crossfit.superglue_wsc import Superglue_Wsc
from iz.crossfit.swag import Swag
from iz.crossfit.tab_fact import TabFact
from iz.crossfit.trec import TREC
from iz.crossfit.trec_finegrained import TREC_Finegrained
from iz.crossfit.tweet_eval import TweetEval
from iz.crossfit.tweet_qa import TweetQA
from iz.crossfit.web_questions import WebQuestions
from iz.crossfit.wiki_auto import WikiAuto
from iz.crossfit.wiki_bio import WikiBio
from iz.crossfit.wiki_qa import WikiQA
from iz.crossfit.wiki_split import WikiSplit
from iz.crossfit.wikisql import WikiSQL
from iz.crossfit.winogrande import Winogrande
from iz.crossfit.wiqa import WIQA
from iz.crossfit.xsum import XSum
from iz.crossfit.yahoo_answers_topics import YahooAnswersTopics
from iz.crossfit.yelp_polarity import YelpPolarity
from iz.crossfit.yelp_review_full import YelpReviewFull

TASKS = {
    "acronym_identification": AcronymIdentification(),
    "ade_classification": AdeCorpusV2_Classfication(),
    "ade_dosage": AdeCorpusV2_Dosage(),
    "ade_effect": AdeCorpusV2_Effect(),
    "adversarial_qa": AdversarialQA(),
    "aeslc": AESLC(),
    "agnews": AGNews(),
    "ai2_arc": AI2_ARC(),
    "amazon_polarity": AmazonPolarity(),
    "anli": ANLI(),
    "app_reviews": AppReviews(),
    "aqua_rat": AquaRat(),
    "art": ART(),
    "aslg_pc12": ASLG_PC12(),
    "biomrc": BioMRC(),
    "boolq": BoolQ(),
    "circa": Circa(),
    "climate_fever": ClimateFever(),
    "codah": CODAH(),
    "commongen": CommonGen(),
    "commonsense_qa": CommonsenseQA(),
    "cos_e": CoS_E(),
    "cosmos_qa": CosmosQA(),
    "crawl_domain": CrawlDomain(),
    "crows_pairs": CrowsPairs(),
    "dbpedia_14": DBpedia14(),
    "definite_pronoun_resolution": DefinitePronounResolution(),
    "discovery": Discovery(),
    "dream": Dream(),
    "duorc": DuoRC(),
    "e2e_nlg_cleaned": E2E_NLG(),
    "emo": Emo(),
    "emotion": Emotion(),
    "empathetic_dialogues": EmpatheticDialogues(),
    "fewshot_gym_dataset": FewshotGymDataset(),
    "fewshot_gym_classification_dataset": FewshotGymClassificationDataset(),
    "fewshot_gym_t2t_dataset": FewshotGymTextToTextDataset(),
    "financial_phrasebank": FinancialPhrasebank(),
    "freebase_qa": FreebaseQA(),
    "gigaword": Gigaword(),
    "glue_cola": Glue_Cola(),
    "glue_mnli": Glue_MNLI(),
    "glue_mrpc": Glue_MRPC(),
    "glue_qnli": Glue_QNLI(),
    "glue_qqp": Glue_QQP(),
    "glue_rte": Glue_RTE(),
    "glue_sst2": Glue_SST2(),
    "glue_wnli": Glue_WNLI(),
    "google_wellformed_query": GoogleWellformedQuery(),
    "hate_speech18": HateSpeech18(),
    "hate_speech_offensive": HateSpeechOffensive(),
    "hatexplain": HatExplain(),
    "health_fact": HealthFact(),
    "hellaswag": HellaSwag(),
    "hotpot_qa": HotpotQA(),
    "imdb": IMDB(),
    "jeopardy": Jeopardy(),
    "kilt_ay2": Kilt_AY2(),
    "kilt_fever": Kilt_Fever(),
    "kilt_hotpotqa": kilt_hotpotqa.Kilt_NQ(),
    "kilt_nq": kilt_nq.Kilt_NQ(),
    "kilt_trex": Kilt_TREX(),
    "kilt_wow": Kilt_WoW(),
    "kilt_zsre": Kilt_ZSRE(),
    "liar": Liar(),
    "limit": Limit(),
    "math_qa": MathQA(),
    "mc_taco": MC_TACO(),
    "medical_questions_pairs": MedicalQuestionPairs(),
    "mocha": Mocha(),
    "multi_news": MultiNews(),
    "numer_sense": NumerSense(),
    "onestop_english": OneStopEnglish(),
    "openbookqa": OpenbookQA(),
    "paws": PAWS(),
    "piqa": PIQA(),
    "poem_sentiment": PoemSentiment(),
    "proto_qa": ProtoQA(),
    "qa_srl": QA_SRL(),
    "qasc": QASC(),
    "quail": QUAIL(),
    "quarel": QUAREL(),
    "quartz-with_knowledge": Quartz("with_knowledge"),
    "quartz-no_knowledge": Quartz("no_knowledge"),
    "quoref": Quoref(),
    "reddit_tifu-tldr": Reddit_TIFU("tldr"),
    "reddit_tifu-title": Reddit_TIFU("title"),
    "ropes": ROPES(),
    "rotten_tomatoes": RottenTomatos(),
    "samsum": SAMSum(),
    "scicite": SciCite(),
    "sciq": SciQ(),
    "scitail": SciTail(),
    "search_qa": SearchQA(),
    "sick": Sick(),
    "sms_spam": SMS_Spam(),
    "social_i_qa": SocialIQA(),
    "spider": Spider(),
    "squad-with_context": SQuAD("with_context"),
    "squad-no_context": SQuAD("no_context"),
    "superglue_cb": Superglue_CB(),
    "superglue_copa": Superglue_COPA(),
    "superglue_multirc": Superglue_MultiRC(),
    "superglue_record": Superglue_Record(),
    "superglue_rte": Superglue_RTE(),
    "superglue_wic": Superglue_Wic(),
    "superglue_wsc": Superglue_Wsc(),
    "swag": Swag(),
    "tab_fact": TabFact(),
    "trec": TREC(),
    "trec_finegrained": TREC_Finegrained(),
    "tweet_qa": TweetQA(),
    "web_questions": WebQuestions(),
    "wiki_auto": WikiAuto(),
    "wiki_bio": WikiBio(),
    "wiki_qa": WikiQA(),
    "wiki_split": WikiSplit(),
    "wikisql": WikiSQL(),
    "winogrande": Winogrande(),
    "wiqa": WIQA(),
    "xsum": XSum(),
    "yahoo_answers_topics": YahooAnswersTopics(),
    "yelp_polarity": YelpPolarity(),
    "yelp_review_full": YelpReviewFull(),
}

for config_name in datasets.get_dataset_config_names('blimp'):
    TASKS["blimp-" + config_name] = BLIMP(config_name)
for config_name in datasets.get_dataset_config_names('break_data'):
    TASKS["break-" + config_name] = Break(config_name)
for config_name in datasets.get_dataset_config_names('lama'):
    TASKS["lama-" + config_name] = LAMA(config_name)
for config_name in datasets.get_dataset_config_names('race'):
    TASKS["race-" + config_name] = Race(config_name)
for config_name in datasets.get_dataset_config_names('tweet_eval'):
    TASKS["tweet_eval-" + config_name] = TweetEval(config_name)
for dimension in ["directed_vs_generalized", "disability", "gender", "national_origin", "race", "religion", "sexual_orientation"]:
    TASKS["ethos-" + dimension] = Ethos(dimension)
for subreddit in ["eli5", "asks", "askh"]:
    TASKS["eli5-" + subreddit] = ELI5(subreddit)