questions = [
    "1. In general, how would your best friend describe you as a risk taker?",
    "   a) A real gambler\n   b) Willing to take risks after completing adequate research\n   c) Cautious\n   d) A real risk avoider",
    "2. You are on a TV game show and can choose one of the following. Which would you take?",
    "   a) $1,000 in cash\n   b) A 50% chance at winning $5,000\n   c) A 25% chance at winning $10,000\n   d) A 5% chance at winning $100,000",
    "3. You have just finished saving for a “once-in-a-lifetime” vacation. Three weeks before you leave, you lose your job. You would:",
    "   a) Cancel the vacation\n   b) Take a much more modest vacation\n   c) Go as scheduled (need the break to job hunt)\n   d) Extend the vacation — it might be your last chance to go first-class",
    "5. If you unexpectedly received $20,000 to invest, what would you do?",
    "   a) Deposit it in a bank account, money market, or insured CD\n   b) Invest in safe high-quality bonds or bond funds\n   c) Invest in stocks or stock mutual funds",
    "6. In terms of experience, how comfortable are you investing in stocks or stock mutual funds?",
    "   a) Not at all comfortable\n   b) Somewhat comfortable\n   c) Very comfortable",
    "8. When you think of the word “risk” which of the following words comes to mind first?",
    "   a) Loss\n   b) Uncertainty\n   c) Opportunity\n   d) Thrill",
    "12. Most of your investments are in safe government bonds. Experts predict hard assets (gold, real estate, etc.) will rise while bonds may fall. What would you do?",
    "   a) Hold the bonds\n   b) Sell bonds → half money-market, half hard assets\n   c) Sell all bonds → 100% hard assets\n   d) Sell all bonds, buy hard assets, and borrow more to buy even more",
    "14. Given the best/worst case returns below, which would you prefer?",
    "   a) +$200 / $0\n   b) +$800 / -$200\n   c) +$2,600 / -$800\n   d) +$4,800 / -$2,400",
    "16. You have been given $1,000 extra. Choose between:",
    "   a) A sure gain of $500\n   b) 50% chance to gain $1,000 (50% chance nothing)",
    "17. You have been given $2,000 extra. Choose between:",
    "   a) A sure loss of $500\n   b) 50% chance to lose $1,000 (50% chance lose nothing)",
    "18. You inherit $100,000 but must invest it ALL in exactly ONE of the following:",
    "   a) Savings account or money-market fund\n   b) Mutual fund of stocks & bonds\n   c) Portfolio of 15 common stocks\n   d) Commodities (gold, silver, oil)",
    "19. If you had to invest $20,000, which allocation feels most appealing?",
    "   a) 60% low-risk, 30% medium-risk, 10% high-risk\n   b) 30% low-risk, 40% medium-risk, 30% high-risk\n   c) 10% low-risk, 40% medium-risk, 50% high-risk",
    "20. A trusted friend (geologist) offers a gold mine deal: 20% chance of success (50–100× return) or total loss. How much would you invest?",
    "   a) Nothing\n   b) One month’s salary\n   c) Three months’ salary\n   d) Six months’ salary",
]

# Scoring dictionary (index = question number - 1)
scoring = {
    0: {"a": 4, "b": 3, "c": 2, "d": 1},
    1: {"a": 1, "b": 2, "c": 3, "d": 4},
    2: {"a": 1, "b": 2, "c": 3, "d": 4},
    3: {"a": 1, "b": 2, "c": 3},
    4: {"a": 1, "b": 2, "c": 3},
    5: {"a": 1, "b": 2, "c": 3, "d": 4},
    6: {"a": 1, "b": 2, "c": 3, "d": 4},
    7: {"a": 1, "b": 2, "c": 3, "d": 4},
    8: {"a": 1, "b": 3},
    9: {"a": 1, "b": 3},
    10: {"a": 1, "b": 2, "c": 3, "d": 4},
    11: {"a": 1, "b": 2, "c": 3},
    12: {"a": 1, "b": 2, "c": 3, "d": 4},
}


def get_category(score):
    if score <= 25:
        return "Low"
    elif score <= 33:
        return "Moderate"
    elif score >= 33:
        return "High"


print("=== Grable & Lytton 13-Item Financial Risk Tolerance Quiz ===\n")
print(
    "Answer each question by typing only the letter (a, b, c, or d) and press Enter.\n"
)

answers = []
for i in range(13):
    while True:
        print(questions[i * 2])
        print(questions[i * 2 + 1])
        ans = input("\nYour answer (a/b/c/d): ").strip().lower()
        valid_options = list(scoring[i].keys())
        if ans in valid_options:
            answers.append(ans)
            print("-" * 50)
            break
        else:
            print("Invalid choice! Please type only a, b, c, or d from the options.\n")

# Calculate final score
total = sum(scoring[i][answers[i]] for i in range(13))

category = get_category(total)

print("\n" + "=" * 60)
print(f"   YOUR RESULT")
print(f"   Total Score: {total} out of 47")
print(f"   Risk Tolerance: {category}")
print("=" * 60)
