import re
from typing import Any, Dict, Iterator, List

import datasets
from lm_eval.tasks.sciq import SciQ

from catwalk.tasks.crossfit import kilt_hotpotqa, kilt_nq
from catwalk.tasks.crossfit.acronym_identification import AcronymIdentification
from catwalk.tasks.crossfit.ade_classification import AdeCorpusV2_Classfication
from catwalk.tasks.crossfit.ade_dosage import AdeCorpusV2_Dosage
from catwalk.tasks.crossfit.ade_effect import AdeCorpusV2_Effect
from catwalk.tasks.crossfit.adversarial_qa import AdversarialQA
from catwalk.tasks.crossfit.aeslc import AESLC
from catwalk.tasks.crossfit.agnews import AGNews
from catwalk.tasks.crossfit.ai2_arc import AI2_ARC
from catwalk.tasks.crossfit.amazon_polarity import AmazonPolarity
from catwalk.tasks.crossfit.anli import ANLI
from catwalk.tasks.crossfit.app_reviews import AppReviews
from catwalk.tasks.crossfit.aqua_rat import AquaRat
from catwalk.tasks.crossfit.art import ART
from catwalk.tasks.crossfit.aslg_pc12 import ASLG_PC12
from catwalk.tasks.crossfit.biomrc import BioMRC
from catwalk.tasks.crossfit.blimp import BLIMP
from catwalk.tasks.crossfit.boolq import BoolQ
from catwalk.tasks.crossfit.break_task import Break
from catwalk.tasks.crossfit.circa import Circa
from catwalk.tasks.crossfit.climate_fever import ClimateFever
from catwalk.tasks.crossfit.codah import CODAH
from catwalk.tasks.crossfit.commongen import CommonGen
from catwalk.tasks.crossfit.commonsense_qa import CommonsenseQA
from catwalk.tasks.crossfit.cos_e import CoS_E
from catwalk.tasks.crossfit.cosmos_qa import CosmosQA
from catwalk.tasks.crossfit.crawl_domain import CrawlDomain
from catwalk.tasks.crossfit.crows_pairs import CrowsPairs
from catwalk.tasks.crossfit.dbpedia_14 import DBpedia14
from catwalk.tasks.crossfit.definite_pronoun_resolution import DefinitePronounResolution
from catwalk.tasks.crossfit.discovery import Discovery
from catwalk.tasks.crossfit.dream import Dream
from catwalk.tasks.crossfit.duorc import DuoRC
from catwalk.tasks.crossfit.e2e_nlg_cleaned import E2E_NLG
from catwalk.tasks.crossfit.eli5 import ELI5
from catwalk.tasks.crossfit.emo import Emo
from catwalk.tasks.crossfit.emotion import Emotion
from catwalk.tasks.crossfit.empathetic_dialogues import EmpatheticDialogues
from catwalk.tasks.crossfit.ethos import Ethos
from catwalk.tasks.crossfit.fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset, \
    FewshotGymTextToTextDataset
from catwalk.tasks.crossfit.financial_phrasebank import FinancialPhrasebank
from catwalk.tasks.crossfit.freebase_qa import FreebaseQA
from catwalk.tasks.crossfit.gigaword import Gigaword
from catwalk.tasks.crossfit.glue_cola import Glue_Cola
from catwalk.tasks.crossfit.glue_mnli import Glue_MNLI
from catwalk.tasks.crossfit.glue_mrpc import Glue_MRPC
from catwalk.tasks.crossfit.glue_qnli import Glue_QNLI
from catwalk.tasks.crossfit.glue_qqp import Glue_QQP
from catwalk.tasks.crossfit.glue_rte import Glue_RTE
from catwalk.tasks.crossfit.glue_sst2 import Glue_SST2
from catwalk.tasks.crossfit.glue_wnli import Glue_WNLI
from catwalk.tasks.crossfit.google_wellformed_query import GoogleWellformedQuery
from catwalk.tasks.crossfit.hate_speech18 import HateSpeech18
from catwalk.tasks.crossfit.hate_speech_offensive import HateSpeechOffensive
from catwalk.tasks.crossfit.hatexplain import HatExplain
from catwalk.tasks.crossfit.health_fact import HealthFact
from catwalk.tasks.crossfit.hellaswag import HellaSwag
from catwalk.tasks.crossfit.hotpot_qa import HotpotQA
from catwalk.tasks.crossfit.imdb import IMDB
from catwalk.tasks.crossfit.jeopardy import Jeopardy
from catwalk.tasks.crossfit.kilt_ay2 import Kilt_AY2
from catwalk.tasks.crossfit.kilt_fever import Kilt_Fever
from catwalk.tasks.crossfit.kilt_trex import Kilt_TREX
from catwalk.tasks.crossfit.kilt_wow import Kilt_WoW
from catwalk.tasks.crossfit.kilt_zsre import Kilt_ZSRE
from catwalk.tasks.crossfit.lama import LAMA
from catwalk.tasks.crossfit.liar import Liar
from catwalk.tasks.crossfit.limit import Limit
from catwalk.tasks.crossfit.math_qa import MathQA
from catwalk.tasks.crossfit.mc_taco import MC_TACO
from catwalk.tasks.crossfit.medical_questions_pairs import MedicalQuestionPairs
from catwalk.tasks.crossfit.mocha import Mocha
from catwalk.tasks.crossfit.multi_news import MultiNews
from catwalk.tasks.crossfit.numer_sense import NumerSense
from catwalk.tasks.crossfit.onestop_english import OneStopEnglish
from catwalk.tasks.crossfit.openbookqa import OpenbookQA
from catwalk.tasks.crossfit.paws import PAWS
from catwalk.tasks.crossfit.piqa import PIQA
from catwalk.tasks.crossfit.poem_sentiment import PoemSentiment
from catwalk.tasks.crossfit.proto_qa import ProtoQA
from catwalk.tasks.crossfit.qa_srl import QA_SRL
from catwalk.tasks.crossfit.qasc import QASC
from catwalk.tasks.crossfit.quail import QUAIL
from catwalk.tasks.crossfit.quarel import QUAREL
from catwalk.tasks.crossfit.quartz import Quartz
from catwalk.tasks.crossfit.quoref import Quoref
from catwalk.tasks.crossfit.race import Race
from catwalk.tasks.crossfit.reddit_tifu import Reddit_TIFU
from catwalk.tasks.crossfit.ropes import ROPES
from catwalk.tasks.crossfit.rotten_tomatoes import RottenTomatos
from catwalk.tasks.crossfit.samsum import SAMSum
from catwalk.tasks.crossfit.scicite import SciCite
from catwalk.tasks.crossfit.scitail import SciTail
from catwalk.tasks.crossfit.search_qa import SearchQA
from catwalk.tasks.crossfit.sick import Sick
from catwalk.tasks.crossfit.sms_spam import SMS_Spam
from catwalk.tasks.crossfit.social_i_qa import SocialIQA
from catwalk.tasks.crossfit.spider import Spider
from catwalk.tasks.crossfit.squad import SQuAD
from catwalk.tasks.crossfit.superglue_cb import Superglue_CB
from catwalk.tasks.crossfit.superglue_copa import Superglue_COPA
from catwalk.tasks.crossfit.superglue_multirc import Superglue_MultiRC
from catwalk.tasks.crossfit.superglue_record import Superglue_Record
from catwalk.tasks.crossfit.superglue_rte import Superglue_RTE
from catwalk.tasks.crossfit.superglue_wic import Superglue_Wic
from catwalk.tasks.crossfit.superglue_wsc import Superglue_Wsc
from catwalk.tasks.crossfit.swag import Swag
from catwalk.tasks.crossfit.tab_fact import TabFact
from catwalk.tasks.crossfit.trec import TREC
from catwalk.tasks.crossfit.trec_finegrained import TREC_Finegrained
from catwalk.tasks.crossfit.tweet_eval import TweetEval
from catwalk.tasks.crossfit.tweet_qa import TweetQA
from catwalk.tasks.crossfit.web_questions import WebQuestions
from catwalk.tasks.crossfit.wiki_auto import WikiAuto
from catwalk.tasks.crossfit.wiki_bio import WikiBio
from catwalk.tasks.crossfit.wiki_qa import WikiQA
from catwalk.tasks.crossfit.wiki_split import WikiSplit
from catwalk.tasks.crossfit.wikisql import WikiSQL
from catwalk.tasks.crossfit.winogrande import Winogrande
from catwalk.tasks.crossfit.wiqa import WIQA
from catwalk.tasks.crossfit.xsum import XSum
from catwalk.tasks.crossfit.yahoo_answers_topics import YahooAnswersTopics
from catwalk.tasks.crossfit.yelp_polarity import YelpPolarity
from catwalk.tasks.crossfit.yelp_review_full import YelpReviewFull

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


def get_data_as_dicts(task: FewshotGymDataset, split: str, fields: List[str]) -> Iterator[Dict[str, Any]]:
    dataset = task.load_dataset()
    train_data, test_data = task.get_train_test_lines(dataset)
    if split == "train":
        data = train_data
    elif split == "validation":
        data = test_data  # For some reason, CrossFit calls this "test", but the code loads the validation set.
    else:
        raise KeyError()
    del train_data
    del test_data

    for input, output in data:
        result = {
            "target": output
        }

        input = " " + input
        field_values = []
        for next_field_name in fields:
            field_value, input = input.split(f" {next_field_name}:", 1)
            field_values.append(field_value.replace("[SEP]", "").strip())
        field_values.append(input.replace("[SEP]", "").strip())
        assert len(field_values) == len(fields) + 1

        field_zero = field_values.pop(0)
        if len(field_zero) > 0:
            result["field_0"] = field_zero

        for field_name, field_value in zip(fields, field_values):
            result[field_name] = field_value
        yield result
