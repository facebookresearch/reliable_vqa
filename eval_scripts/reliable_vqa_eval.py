# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#### LICENSE from https://github.com/GT-Vision-Lab/VQA
# Copyright (c) 2014, Aishwarya Agrawal
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and documentation are
# those
# of the authors and should not be interpreted as representing official
# policies,
# either expressed or implied, of the FreeBSD Project.

import re

from collections import OrderedDict

import numpy as np
from sklearn.metrics import auc


contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manualMap = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(?<=\d)(\,)+(?=\d)")
puncStrip = re.compile(
    r"(?<=[ \\;/\"`\[\](){}<>@=+_\-,?!])([\\;/\"`\[\](){}<>@=+_\-,?!])|([\\;/\"`\[\](){}<>@=+_\-,?!])(?=[ \\;/\"`\[\](){}<>@=+_\-,?!])"
)
puncStrip2 = re.compile(r"(?<=[a-zA-Z])([\\;/\"`\[\](){}<>@=+_\-,?!])(?=[a-zA-Z])")
puncStripBegin = re.compile(r"\A([ \\;/\"`\[\](){}<>@=+_\-,?!]+)(?=[a-zA-Z0-9 ])")
puncStripEnd = re.compile(r"(?<=[a-zA-Z0-9 ])([ \\;/\"`\[\](){}<>@=+_\-,?!]+)\Z")
spaceCleanup = re.compile(r"([ ]+)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


class ReliabilityEval:
    def __init__(self, quesIds, risk_tolerances=None, costs=None, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = OrderedDict()
        self.evalThresholdQA = OrderedDict()
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.all_qids = quesIds

        self.risk_tolerances = risk_tolerances
        self.costs = costs

        if self.risk_tolerances is None:
            self.risk_tolerances = [0.01, 0.05, 0.1, 0.2]
        if self.costs is None:
            self.costs = [1, 10, 100]

        self.cost_thresholds = None

        self.fptp_stats = {}

    def evaluate(
        self,
        vqa,
        vqaRes,
        thresholdRes,
        quesIds=None,
        thresholdQuesIds=None,
        skip_reliability=False,
    ):
        if quesIds == None:
            quesIds = self.all_qids

        # =================================================
        # Compute accuracy
        # =================================================
        self.computeAccuracy(vqa, vqaRes, quesIds, is_threshold=False)

        if skip_reliability:
            return

        self.setCatR()

        if thresholdRes is not None:
            assert thresholdQuesIds is not None
            # Set self.evalThresholdQA with accuracy & confidence for each prediction in
            # the threshold set
            self.computeAccuracy(vqa, thresholdRes, thresholdQuesIds, is_threshold=True)

            # Compute thresholds for effective reliability, and update self.cost_thresholds
            self.computeThresholds()
            print("Chosen thresholds for effective reliability:")
            for cost, threshold in self.cost_thresholds.items():
                print("Cost: {}, Threshold: {}".format(cost, threshold))
            # Compute effective reliability scores for each cost value
            self.setPhiAtCosts()

    def computeAccuracy(self, vqa, vqaRes, quesIds=None, is_threshold=False):
        accQA = []
        step = 0

        for quesId in quesIds:
            gt = vqa.qa[quesId]
            res = vqaRes.qa[quesId]

            for ansDic in gt["answers"]:
                ansDic["answer"] = ansDic["answer"].replace("\n", " ")
                ansDic["answer"] = ansDic["answer"].replace("\t", " ")
                ansDic["answer"] = ansDic["answer"].strip()
            resAns = res["answer"]
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()

            resConf = res["confidence"]

            gtAcc = []
            gtAnswers = [ans["answer"] for ans in gt["answers"]]

            if len(set(gtAnswers)) > 1:
                for ansDic in gt["answers"]:
                    ansDic["answer"] = self.processPunctuation(ansDic["answer"])
                    ansDic["answer"] = self.processDigitArticle(ansDic["answer"])
                resAns = self.processPunctuation(resAns)
                resAns = self.processDigitArticle(resAns)

            #######################################################
            for gtAnsDatum in gt["answers"]:
                otherGTAns = [item for item in gt["answers"] if item != gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            #######################################################
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            risk = 1.0 - avgGTAcc
            accQA.append(avgGTAcc)
            ########################################################
            self.setEvalQA(quesId, avgGTAcc, risk, resConf, is_threshold=is_threshold)
            ########################################################

        if not is_threshold:
            self.setAccuracy(accQA)

    def processPunctuation(self, inText):
        outText = puncStripBegin.sub("", inText)
        outText = puncStripEnd.sub("", outText)
        outText = commaStrip.sub("", outText)
        outText = puncStrip.sub(" ", outText)
        outText = spaceCleanup.sub(" ", outText)
        outText = puncStrip2.sub(" ", outText)
        outText = puncStrip2.sub("", outText)
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions:
                outText[wordId] = contractions[word]
        outText = " ".join(outText)
        return outText

    def setAccuracy(self, accQA):
        self.accuracy["vqa_accuracy"] = round(
            100 * float(sum(accQA)) / len(accQA), self.n
        )

    def setRiskCoverage(self, evalQA, best=False):
        total_questions = len(evalQA)

        covered = 0.0
        cum_score = 0.0

        risks = []
        coverages = []

        temp_items = list(evalQA.items())

        for i, (_, qresult) in enumerate(evalQA.items()):
            if i > 0 and not best:
                assert temp_items[i - 1][1]["confidence"] >= qresult["confidence"]

            covered += 1.0
            cum_score += qresult["indiv_risk"]

            curr_risk = cum_score / covered
            curr_cov = covered / total_questions

            qresult["risk"] = curr_risk
            qresult["coverage"] = curr_cov

            risks.append(curr_risk)
            coverages.append(curr_cov)

        auc_score = auc(coverages, risks)

        key = "auc"
        if best:
            key = "best_" + key

        self.accuracy[key] = round(100.0 * auc_score, self.n)

    def computeCatR(self, evalQA, risk_tolerance, best=False):
        total_questions = len(evalQA)

        _, rc_data = zip(*evalQA.items())
        index = total_questions
        while index > 0 and rc_data[index - 1]["risk"] > risk_tolerance:
            index -= 1
        index -= 1

        if (
            -1 < index < (total_questions - 1)
            and rc_data[index]["confidence"] == rc_data[index + 1]["confidence"]
        ):
            while index > -1 and (
                rc_data[index]["confidence"] == rc_data[index + 1]["confidence"]
                or rc_data[index]["risk"] > risk_tolerance
            ):
                index -= 1

        cov = rc_data[index]["coverage"] if index > -1 else 0.0
        threshold = rc_data[index]["confidence"] if index > -1 else 0.0

        catr = {
            "coverage": round(100.0 * cov, self.n),
            "threshold": threshold,
        }

        key = "cov@{}".format(str(risk_tolerance))

        if best:
            key = "best_" + key

        self.accuracy[key] = catr

    def setCatR(self):
        for is_best in (True, False):
            sort_key = "confidence" if not is_best else "accuracy"

            self.evalQA = OrderedDict(
                sorted(self.evalQA.items(), key=lambda x: -x[1][sort_key])
            )

            self.setRiskCoverage(self.evalQA, best=is_best)

            for rt in self.risk_tolerances:
                self.computeCatR(self.evalQA, rt, best=is_best)

        self.evalQA = OrderedDict(
            sorted(self.evalQA.items(), key=lambda x: -x[1]["confidence"])
        )

    def setPhiAtCosts(self):
        sorted_confs, sorted_scores = self.getSortedArraysForPhi(is_threshold=False)
        for cost in self.costs:
            self.computePhiAtCost(sorted_confs, sorted_scores, cost)
            self.computeBestPossiblePhiAtCost(sorted_scores, cost)

    def getSortedArraysForPhi(self, is_threshold=False):
        qa = self.evalThresholdQA if is_threshold else self.evalQA
        qa = OrderedDict(sorted(qa.items(), key=lambda x: -x[1]["confidence"]))
        sorted_confs = [r["confidence"] for r in qa.values()]  # High to low
        sorted_scores = [
            r["accuracy"] / 100 for r in qa.values()
        ]  # Corresponding scores
        sorted_confs = np.array(sorted_confs)
        sorted_scores = np.array(sorted_scores)
        return sorted_confs, sorted_scores

    def computePhiAtCost(self, sorted_confs, sorted_scores, cost, prefix=""):
        threshold = self.cost_thresholds[cost]
        cum_score = 0.0
        acc_score = 0.0
        num_answered = 0
        total_questions = len(sorted_confs)
        for i in range(total_questions):
            if sorted_confs[i] >= threshold:
                # Choose to answer
                acc_score += sorted_scores[i]
                num_answered += 1
                if sorted_scores[i] == 0:
                    cum_score -= cost
                else:
                    cum_score += sorted_scores[i]

        phi = cum_score / total_questions
        cov = num_answered / total_questions
        risk = 1 - (acc_score / max(num_answered, 1.0))

        # Compute phi without option to abstain
        cum_score = 0.0
        for s in sorted_scores:
            # No option to abstain
            if s == 0:
                cum_score -= cost
            else:
                cum_score += s
        phi_no_abstention = cum_score / len(sorted_scores)

        res = {
            "threshold": threshold,
            "phi": round(100.0 * phi, self.n),
            "coverage": round(100.0 * cov, self.n),
            "risk": round(100.0 * risk, self.n),
            "no_abstention_phi": round(100.0 * phi_no_abstention, self.n),
        }
        self.accuracy[f"phi@{str(cost)}"] = res

    def computeBestPossiblePhiAtCost(self, sorted_scores, cost):
        sorted_costs = []
        for s in sorted_scores:
            if s == 0:
                sorted_costs.append(-cost)
            else:
                sorted_costs.append(s)
        sorted_costs = np.array(sorted_costs)
        total_questions = len(sorted_costs)

        # Add up all positive entries of sorted_costs
        sorted_costs = np.array(sorted_costs)
        max_phi = sorted_costs[sorted_costs > 0].sum()
        best_possible_phi = max_phi / total_questions

        # Coverage
        num_answered = (sorted_costs > 0).sum()
        best_coverage = num_answered / total_questions

        # Risk
        # max_phi is the sum of Acc(x) scores on samples where Acc(x) > 0.
        # A perfect model gets Acc(x) = 1 each time, which equals num_answered,
        # giving a risk of 0.
        best_risk = 1 - (max_phi / max(num_answered, 1.0))

        res = {
            "phi": round(100.0 * best_possible_phi, self.n),
            "coverage": round(100.0 * best_coverage, self.n),
            "risk": round(100.0 * best_risk, self.n),
        }
        self.accuracy[f"best_phi@{str(cost)}"] = res

    def computeThresholds(self):
        self.cost_thresholds = {}
        sorted_confs, sorted_scores = self.getSortedArraysForPhi(is_threshold=True)
        for c in self.costs:
            self.cost_thresholds[c] = self.computeThresholdAtCost(
                sorted_confs, sorted_scores, c
            )

    def computeThresholdAtCost(self, sorted_confs, sorted_scores, c):
        sorted_costs = []
        for score in sorted_scores:
            if score == 0.0:
                sorted_costs.append(-c)
            else:
                sorted_costs.append(score)
        sorted_costs = np.array(sorted_costs)

        all_phis = []
        for i in range(len(sorted_confs)):
            phi = sum(sorted_costs[: i + 1])
            all_phis.append(phi)
        all_phis = np.array(all_phis)
        threshold_candidates = np.where(all_phis == all_phis.max())[0]
        threshold_index = threshold_candidates[
            -1
        ]  # Lowest threshold (i.e., most coverage) with max phi
        threshold = sorted_confs[threshold_index]
        return threshold

    def setEvalQA(self, quesId, acc, risk, conf, is_threshold=False):
        qa = self.evalThresholdQA if is_threshold else self.evalQA
        qa[quesId] = {
            "accuracy": round(100.0 * acc, self.n),
            "indiv_risk": risk,
            "confidence": float(conf),
        }
