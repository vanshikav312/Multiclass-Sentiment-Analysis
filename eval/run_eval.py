"""
20-sentence evaluation harness.
Calls the same predict() function the FastAPI backend uses.
Run from the project root:
    python eval/run_eval.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import predict

CASES = [
    # (input_text, expected_emotion, expect_depressed, expect_anxious)
    # -- clear positive --
    ("I am so happy today, everything is going great",              "joy",      False, False),
    ("I love spending time with my family",                         "love",     False, False),
    ("I am proud of what I achieved this week",                     "joy",      False, False),
    ("I had a wonderful day with my friends",                       "joy",      False, False),
    ("Life feels beautiful and full of possibilities",              "joy",      False, False),

    # -- clear depressive --
    ("I feel completely empty and there is no point in anything",   "sad",      True,  False),
    ("I have been crying every day and cannot get out of bed",      "sad",      True,  False),
    ("Nothing brings me joy anymore and I feel hopeless",           "sad",      True,  False),
    ("I have been feeling really down and worthless lately",        "sad",      True,  False),
    ("I am exhausted all the time and do not want to do anything",  "sad",      True,  False),

    # -- clear anxious --
    ("I have been having panic attacks every day",                  None,       False, True),
    ("My heart races and I cannot breathe when I think about it",   None,       False, True),
    ("I feel so anxious I cannot leave the house",                  None,       False, True),
    ("Everything makes me anxious and I am constantly overwhelmed", None,       False, True),
    ("I am constantly scared that something bad will happen",       None,       False, True),

    # -- ambiguous / mixed --
    ("I am not feeling good today because I had a breakup",         None,       None,  None),
    ("I keep overthinking every little thing",                      None,       None,  None),
    ("I am terrified of what might happen tomorrow",                None,       None,  None),
    ("I cannot stop worrying about everything in my life",          None,       None,  None),
    ("I am dreading tomorrow so much I cannot sleep",               None,       None,  None),
]

def run():
    print("\n" + "=" * 90)
    print(f"  {'INPUT':<48} {'EMOTION':<10} {'DEP':<6} {'ANX':<6} {'RESULT'}")
    print("=" * 90)

    correct = total_scored = 0

    for text, exp_emo, exp_dep, exp_anx in CASES:
        result = predict(text, silent=True)

        emo_raw   = result['emotion']['label'].lower()
        dep_flag  = result['depression']['label'] == 'Depressed'
        anx_flag  = result['anxiety']['label'] == 'Anxious'
        emo_conf  = result['emotion']['confidence']

        checks = []
        if exp_emo is not None:
            ok = emo_raw == exp_emo.lower()
            checks.append(ok)
            total_scored += 1
            if ok: correct += 1
        if exp_dep is not None:
            ok = dep_flag == exp_dep
            checks.append(ok)
            total_scored += 1
            if ok: correct += 1
        if exp_anx is not None:
            ok = anx_flag == exp_anx
            checks.append(ok)
            total_scored += 1
            if ok: correct += 1

        status = "PASS" if all(checks) else ("FAIL" if checks else "----")
        dep_str = ("Yes" if dep_flag else "No")
        anx_str = ("Yes" if anx_flag else "No")
        print(f"  {text[:48]:<48} {emo_raw:<10} {dep_str:<6} {anx_str:<6} {status}")

    print("=" * 90)
    if total_scored:
        print(f"  Score: {correct}/{total_scored}  ({correct/total_scored:.0%})")
    print()


if __name__ == '__main__':
    run()
