import datasets
from lm_eval.tasks.sciq import SciQ

from catwalk2.crossfit import kilt_hotpotqa, kilt_nq
from catwalk2.crossfit.acronym_identification import AcronymIdentification
from catwalk2.crossfit.ade_classification import AdeCorpusV2_Classfication
from catwalk2.crossfit.ade_dosage import AdeCorpusV2_Dosage
from catwalk2.crossfit.ade_effect import AdeCorpusV2_Effect
from catwalk2.crossfit.adversarial_qa import AdversarialQA
from catwalk2.crossfit.aeslc import AESLC
from catwalk2.crossfit.agnews import AGNews
from catwalk2.crossfit.ai2_arc import AI2_ARC
from catwalk2.crossfit.amazon_polarity import AmazonPolarity
from catwalk2.crossfit.anli import ANLI
from catwalk2.crossfit.app_reviews import AppReviews
from catwalk2.crossfit.aqua_rat import AquaRat
from catwalk2.crossfit.art import ART
from catwalk2.crossfit.aslg_pc12 import ASLG_PC12
from catwalk2.crossfit.biomrc import BioMRC
from catwalk2.crossfit.blimp import BLIMP
from catwalk2.crossfit.boolq import BoolQ
from catwalk2.crossfit.break_task import Break
from catwalk2.crossfit.circa import Circa
from catwalk2.crossfit.climate_fever import ClimateFever
from catwalk2.crossfit.codah import CODAH
from catwalk2.crossfit.commongen import CommonGen
from catwalk2.crossfit.commonsense_qa import CommonsenseQA
from catwalk2.crossfit.cos_e import CoS_E
from catwalk2.crossfit.cosmos_qa import CosmosQA
from catwalk2.crossfit.crawl_domain import CrawlDomain
from catwalk2.crossfit.crows_pairs import CrowsPairs
from catwalk2.crossfit.dbpedia_14 import DBpedia14
from catwalk2.crossfit.definite_pronoun_resolution import DefinitePronounResolution
from catwalk2.crossfit.discovery import Discovery
from catwalk2.crossfit.dream import Dream
from catwalk2.crossfit.duorc import DuoRC
from catwalk2.crossfit.e2e_nlg_cleaned import E2E_NLG
from catwalk2.crossfit.eli5 import ELI5
from catwalk2.crossfit.emo import Emo
from catwalk2.crossfit.emotion import Emotion
from catwalk2.crossfit.empathetic_dialogues import EmpatheticDialogues
from catwalk2.crossfit.ethos import Ethos
from catwalk2.crossfit.fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset, \
    FewshotGymTextToTextDataset
from catwalk2.crossfit.financial_phrasebank import FinancialPhrasebank
from catwalk2.crossfit.freebase_qa import FreebaseQA
from catwalk2.crossfit.gigaword import Gigaword
from catwalk2.crossfit.glue_cola import Glue_Cola
from catwalk2.crossfit.glue_mnli import Glue_MNLI
from catwalk2.crossfit.glue_mrpc import Glue_MRPC
from catwalk2.crossfit.glue_qnli import Glue_QNLI
from catwalk2.crossfit.glue_qqp import Glue_QQP
from catwalk2.crossfit.glue_rte import Glue_RTE
from catwalk2.crossfit.glue_sst2 import Glue_SST2
from catwalk2.crossfit.glue_wnli import Glue_WNLI
from catwalk2.crossfit.google_wellformed_query import GoogleWellformedQuery
from catwalk2.crossfit.hate_speech18 import HateSpeech18
from catwalk2.crossfit.hate_speech_offensive import HateSpeechOffensive
from catwalk2.crossfit.hatexplain import HatExplain
from catwalk2.crossfit.health_fact import HealthFact
from catwalk2.crossfit.hellaswag import HellaSwag
from catwalk2.crossfit.hotpot_qa import HotpotQA
from catwalk2.crossfit.imdb import IMDB
from catwalk2.crossfit.jeopardy import Jeopardy
from catwalk2.crossfit.kilt_ay2 import Kilt_AY2
from catwalk2.crossfit.kilt_fever import Kilt_Fever
from catwalk2.crossfit.kilt_trex import Kilt_TREX
from catwalk2.crossfit.kilt_wow import Kilt_WoW
from catwalk2.crossfit.kilt_zsre import Kilt_ZSRE
from catwalk2.crossfit.lama import LAMA
from catwalk2.crossfit.liar import Liar
from catwalk2.crossfit.limit import Limit
from catwalk2.crossfit.math_qa import MathQA
from catwalk2.crossfit.mc_taco import MC_TACO
from catwalk2.crossfit.medical_questions_pairs import MedicalQuestionPairs
from catwalk2.crossfit.mocha import Mocha
from catwalk2.crossfit.multi_news import MultiNews
from catwalk2.crossfit.numer_sense import NumerSense
from catwalk2.crossfit.onestop_english import OneStopEnglish
from catwalk2.crossfit.openbookqa import OpenbookQA
from catwalk2.crossfit.paws import PAWS
from catwalk2.crossfit.piqa import PIQA
from catwalk2.crossfit.poem_sentiment import PoemSentiment
from catwalk2.crossfit.proto_qa import ProtoQA
from catwalk2.crossfit.qa_srl import QA_SRL
from catwalk2.crossfit.qasc import QASC
from catwalk2.crossfit.quail import QUAIL
from catwalk2.crossfit.quarel import QUAREL
from catwalk2.crossfit.quartz import Quartz
from catwalk2.crossfit.quoref import Quoref
from catwalk2.crossfit.race import Race
from catwalk2.crossfit.reddit_tifu import Reddit_TIFU
from catwalk2.crossfit.ropes import ROPES
from catwalk2.crossfit.rotten_tomatoes import RottenTomatos
from catwalk2.crossfit.samsum import SAMSum
from catwalk2.crossfit.scicite import SciCite
from catwalk2.crossfit.scitail import SciTail
from catwalk2.crossfit.search_qa import SearchQA
from catwalk2.crossfit.sick import Sick
from catwalk2.crossfit.sms_spam import SMS_Spam
from catwalk2.crossfit.social_i_qa import SocialIQA
from catwalk2.crossfit.spider import Spider
from catwalk2.crossfit.squad import SQuAD
from catwalk2.crossfit.superglue_cb import Superglue_CB
from catwalk2.crossfit.superglue_copa import Superglue_COPA
from catwalk2.crossfit.superglue_multirc import Superglue_MultiRC
from catwalk2.crossfit.superglue_record import Superglue_Record
from catwalk2.crossfit.superglue_rte import Superglue_RTE
from catwalk2.crossfit.superglue_wic import Superglue_Wic
from catwalk2.crossfit.superglue_wsc import Superglue_Wsc
from catwalk2.crossfit.swag import Swag
from catwalk2.crossfit.tab_fact import TabFact
from catwalk2.crossfit.trec import TREC
from catwalk2.crossfit.trec_finegrained import TREC_Finegrained
from catwalk2.crossfit.tweet_eval import TweetEval
from catwalk2.crossfit.tweet_qa import TweetQA
from catwalk2.crossfit.web_questions import WebQuestions
from catwalk2.crossfit.wiki_auto import WikiAuto
from catwalk2.crossfit.wiki_bio import WikiBio
from catwalk2.crossfit.wiki_qa import WikiQA
from catwalk2.crossfit.wiki_split import WikiSplit
from catwalk2.crossfit.wikisql import WikiSQL
from catwalk2.crossfit.winogrande import Winogrande
from catwalk2.crossfit.wiqa import WIQA
from catwalk2.crossfit.xsum import XSum
from catwalk2.crossfit.yahoo_answers_topics import YahooAnswersTopics
from catwalk2.crossfit.yelp_polarity import YelpPolarity
from catwalk2.crossfit.yelp_review_full import YelpReviewFull

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