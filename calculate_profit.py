import csv
import re
import os
import tempfile
import sys
from datetime import datetime

FILENAME = 'goisheet - Sheet3 (1).csv'
NIGHTMARE_FILE = 'goisheet - nightmare matches.csv'

# Helper to normalize strings: lowercase, remove spaces and punctuation
def normalize(s):
    if not s:
        return ''
    return re.sub(r'[^a-z0-9]', '', s.lower())

def extract_year_month(date_str):
    if not date_str:
        return ''
    m = re.search(r'(\d{4})[-/](\d{2})', date_str)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m")
        except Exception:
            continue
    m2 = re.match(r'([A-Za-z]+)\s+(\d{4})', date_str)
    if m2:
        try:
            dt = datetime.strptime(f"{m2.group(2)}-{m2.group(1)}", "%Y-%B")
            return dt.strftime("%Y-%m")
        except Exception:
            try:
                dt = datetime.strptime(f"{m2.group(2)}-{m2.group(1)}", "%Y-%b")
                return dt.strftime("%Y-%m")
            except Exception:
                return ''
    return ''

def load_nightmare_matches():
    nightmare_set = set()
    with open(NIGHTMARE_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n1 = normalize(row['Name 1'])
            n2 = normalize(row['Name 2'])
            company = normalize(row['Company'])
            month = row['Month']
            # Ignore rows where both names are empty
            if not n1 and not n2:
                continue
            # Add both (n1, n2) and (n2, n1) for symmetry
            if n1 and n2:
                nightmare_set.add((n1, n2, company, month))
                nightmare_set.add((n2, n1, company, month))
            elif n2:  # single-word match (n2 is the word)
                nightmare_set.add((n2, '', company, month))
    return nightmare_set

def any_word_match(name1, name2, company, month, nightmare_set):
    ignore_words = {'mark'}
    weak_words = {'van', 'der', 'el'}
    words1 = [w for w in re.findall(r'\w+', name1.lower()) if w not in ignore_words]
    words2 = [w for w in re.findall(r'\w+', name2.lower()) if w not in ignore_words]
    n1 = normalize(name1)
    n2 = normalize(name2)
    company_norm = normalize(company)
    # Exclude nightmare matches
    match_words = set()
    for w1 in words1:
        for w2 in words2:
            if w1 == w2:
                # Check for explicit pair block
                if ((n1, n2, company_norm, month) in nightmare_set or
                    (n2, n1, company_norm, month) in nightmare_set or
                    (w1, '', company_norm, month) in nightmare_set or
                    (w2, '', company_norm, month) in nightmare_set):
                    continue
                match_words.add(w1)
    # If all matching words are weak, do not match
    if match_words and all(w in weak_words for w in match_words):
        return False
    return len(match_words) > 0

def loose_match(row1, row2, nightmare_set):
    company1 = normalize(row1['Company'])
    company2 = normalize(row2['Company'])
    period1 = extract_year_month(row1['Payment Cycle Period (Calculated)'])
    period2 = extract_year_month(row2['Payment Cycle Period (Calculated)'])
    name1 = row1['Name']
    name2 = row2['Name']
    if company1 == company2 and period1 == period2:
        return any_word_match(name1, name2, row1['Company'], period1, nightmare_set)
    return False

def is_revenue_row(row):
    return 'revenue' in normalize(row.get('Bucket', ''))

def print_fred_costs_debug():
    with open(FILENAME, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Find Fred's revenue row for March 2025 using loose matching on new field
    target = {'Name': "Fred M'Bikay", 'Company': "Rituals", 'Payment Cycle Period (Calculated)': "2025-03"}
    target_row = None
    for i, row in enumerate(rows):
        if is_revenue_row(row) and loose_match(target, row, set()): # Pass an empty set for nightmare_set
            target_row = (i, row)
            break
    if not target_row:
        print("Fred M'Bikay revenue row not found.")
        return
    idx, row = target_row
    # Find matching cost rows above
    matches = []
    for j in range(idx):
        candidate = rows[j]
        if not is_revenue_row(candidate) and loose_match(row, candidate, set()): # Pass an empty set for nightmare_set
            matches.append((j+2, candidate))  # +2 for 1-based and header
    print(f"Matched cost rows for Fred M'Bikay, March 2025 (row {idx+2}):")
    for rownum, match in matches:
        print(f"Row {rownum}: {match}")
    print(f"Total matched: {len(matches)}")
    print(f"Sum of costs: {sum(float(m['Payment Amount (payment currency)']) if m['Payment Amount (payment currency)'] else 0 for _, m in matches):.2f}")

def print_shareen_may2025_debug():
    with open(FILENAME, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Find Shareen's revenue row for May 2025 using loose matching
    target = {'Name': "Shareen", 'Company': "Stone & Bridges", 'Payment Cycle Period (Calculated)': "May 2025"}
    target_row = None
    for i, row in enumerate(rows):
        if is_revenue_row(row) and loose_match(target, row, set()): # Pass an empty set for nightmare_set
            target_row = (i, row)
            break
    if not target_row:
        print("Shareen revenue row for May 2025 not found.")
        return
    idx, row = target_row
    # Find matching cost rows above
    matches = []
    for j in range(idx):
        candidate = rows[j]
        if not is_revenue_row(candidate) and loose_match(row, candidate, set()): # Pass an empty set for nightmare_set
            matches.append((j+2, candidate))  # +2 for 1-based and header
    print(f"Matched cost rows for Shareen, May 2025 (row {idx+2}):")
    for rownum, match in matches:
        print(f"Row {rownum}: {match}")
    print(f"Total matched: {len(matches)}")
    print(f"Sum of costs: {sum(float(m['Payment Amount (payment currency)']) if m['Payment Amount (payment currency)'] else 0 for _, m in matches):.2f}")

def debug_mark_jacobs_may2024():
    target_name = 'Mark Jacobs'
    target_month = '2024-05'
    with open(FILENAME, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    found = False
    for i, row in enumerate(reader):
        if is_revenue_row(row):
            # Check name and month match
            if any_word_match(target_name, row['Name'], row['Company'], extract_year_month(row['Payment Cycle Period (Calculated)']), set()): # Pass an empty set for nightmare_set
                print(f"Revenue row found at line {i+2} (including header):")
                print(row)
                found = True
                # Find matching cost rows above
                matches = []
                for j in range(i):
                    cost_row = reader[j]
                    if not is_revenue_row(cost_row) and loose_match(row, cost_row, set()): # Pass an empty set for nightmare_set
                        matches.append((j+2, cost_row))
                if matches:
                    print(f"\nMatching cost rows used in profit calculation:")
                    for line_num, match_row in matches:
                        print(f"Row {line_num}: {match_row}")
                else:
                    print("No matching cost rows found.")
                return
    if not found:
        print("No revenue row found for Mark Jacobs in May 2024.")

def find_nightmare_matches():
    import csv as pycsv
    with open(FILENAME, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    ignore_words = {'mark'}
    seen_pairs = set()
    results = []
    nightmare_set = load_nightmare_matches()
    for i, row1 in enumerate(rows):
        name1 = row1['Name'].strip()
        company1 = normalize(row1['Company'])
        month1 = extract_year_month(row1['Payment Cycle Period (Calculated)'])
        words1 = set([w for w in re.findall(r'\w+', name1.lower()) if w not in ignore_words])
        for j, row2 in enumerate(rows):
            if i >= j:
                continue
            name2 = row2['Name'].strip()
            company2 = normalize(row2['Company'])
            month2 = extract_year_month(row2['Payment Cycle Period (Calculated)'])
            words2 = set([w for w in re.findall(r'\w+', name2.lower()) if w not in ignore_words])
            if name1.lower() != name2.lower() and words1 & words2:
                if len(words1 & words2) == 1 and company1 == company2 and month1 == month2:
                    pair = tuple(sorted([name1.lower(), name2.lower(), company1, month1]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        common_word = list(words1 & words2)[0]
                        results.append({
                            'Name 1': name1,
                            'Name 2': name2,
                            'Common Word': common_word,
                            'Company': row1['Company'],
                            'Month': month1
                        })
    # Write to CSV
    with open('nightmare_matches.csv', 'w', newline='', encoding='utf-8') as outcsv:
        writer = pycsv.DictWriter(outcsv, fieldnames=['Name 1', 'Name 2', 'Common Word', 'Company', 'Month'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} nightmare matches to nightmare_matches.csv")

def parse_payment_amount(val):
    if not val:
        return 0.0
    s = str(val).replace(' ', '')
    if '€' in s:
        s = s.replace('€', '').replace('.', '').replace(',', '.')
    else:
        s = s.replace(',', '')
    try:
        return float(s)
    except Exception:
        return 0.0

def main():
    nightmare_set = load_nightmare_matches()
    if len(sys.argv) > 1 and sys.argv[1] == '--shareen-may2025-debug':
        print_shareen_may2025_debug()
        return
    if len(sys.argv) > 1 and sys.argv[1] == '--fred-costs-debug':
        print_fred_costs_debug()
        return
    with open(FILENAME, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        fieldnames = reader[0].keys()

    # Remove all existing profit rows
    original_rows = [row for row in reader if row.get('Bucket', '').strip().lower() != 'profit']

    new_rows = []
    for i, row in enumerate(original_rows):
        if is_revenue_row(row):
            payment = parse_payment_amount(row.get('Payment Amount (payment currency)', ''))
            # If payment is 0, output n/a
            if payment == 0:
                profit_value = 'n/a'
            else:
                # Find matching cost rows above
                cost_sum = 0.0
                cost_count = 0
                for j in range(i):
                    cost_row = original_rows[j]
                    if not is_revenue_row(cost_row) and loose_match(row, cost_row, nightmare_set):
                        cost_sum += parse_payment_amount(cost_row.get('Payment Amount (payment currency)', ''))
                        cost_count += 1
                if cost_count == 0:
                    profit_value = 'n/a'
                else:
                    profit_value = payment - cost_sum
            # Create profit row
            profit_row = dict(row)
            profit_row['Bucket'] = 'profit'
            profit_row['Payment Amount (payment currency)'] = profit_value
            new_rows.append(profit_row)
    # Write back to file
    with open(FILENAME, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(original_rows)
        writer.writerows(new_rows)

def debug_roos_june2025():
    target_name = 'Roos van der Pluijm'
    target_month = '2025-06'
    nightmare_set = load_nightmare_matches()
    with open(FILENAME, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    found = False
    for i, row in enumerate(reader):
        if is_revenue_row(row):
            # Check name and month match
            if any_word_match(target_name, row['Name'], row['Company'], extract_year_month(row['Payment Cycle Period (Calculated)']), nightmare_set) and extract_year_month(row['Payment Cycle Period (Calculated)']) == target_month:
                print(f"Revenue row found at line {i+2} (including header):")
                print(row)
                found = True
                # Find matching cost rows above
                matches = []
                for j in range(i):
                    cost_row = reader[j]
                    if not is_revenue_row(cost_row) and loose_match(row, cost_row, nightmare_set):
                        matches.append((j+2, cost_row))
                if matches:
                    print(f"\nMatching cost rows used in profit calculation:")
                    for line_num, match_row in matches:
                        print(f"Row {line_num}: {match_row}")
                else:
                    print("No matching cost rows found.")
                return
    if not found:
        print("No revenue row found for Roos van der Pluijm in June 2025.")

def debug_roos_june2025_verbose():
    target_name = 'Roos van der Pluijm'
    target_month = '2025-06'
    nightmare_set = load_nightmare_matches()
    with open(FILENAME, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    found = False
    for i, row in enumerate(reader):
        if is_revenue_row(row):
            # Check name and month match
            if any_word_match(target_name, row['Name'], row['Company'], extract_year_month(row['Payment Cycle Period (Calculated)']), nightmare_set) and extract_year_month(row['Payment Cycle Period (Calculated)']) == target_month:
                print(f"Revenue row found at line {i+2} (including header):")
                print(row)
                found = True
                # Find matching cost rows above
                matches = []
                for j in range(i):
                    cost_row = reader[j]
                    if not is_revenue_row(cost_row):
                        # Show matching logic
                        company1 = normalize(row['Company'])
                        company2 = normalize(cost_row['Company'])
                        period1 = extract_year_month(row['Payment Cycle Period (Calculated)'])
                        period2 = extract_year_month(cost_row['Payment Cycle Period (Calculated)'])
                        name1 = row['Name']
                        name2 = cost_row['Name']
                        if company1 == company2 and period1 == period2:
                            match = any_word_match(name1, name2, row['Company'], period1, nightmare_set)
                            reason = 'MATCH' if match else 'BLOCKED (nightmare or no word match)'
                        else:
                            match = False
                            reason = 'BLOCKED (company/month mismatch)'
                        print(f"Row {j+2}: {cost_row['Name']} | Company: {cost_row['Company']} | Month: {period2} | Reason: {reason}")
                        if match:
                            matches.append((j+2, cost_row))
                if matches:
                    print(f"\nMatching cost rows used in profit calculation:")
                    for line_num, match_row in matches:
                        print(f"Row {line_num}: {match_row}")
                else:
                    print("No matching cost rows found.")
                return
    if not found:
        print("No revenue row found for Roos van der Pluijm in June 2025.")

def trace_roos_june2025():
    target_name = 'Roos van der Pluijm'
    target_month = '2025-06'
    nightmare_set = load_nightmare_matches()
    with open(FILENAME, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    found = False
    for i, row in enumerate(reader):
        if is_revenue_row(row):
            # Check name and month match
            if any_word_match(target_name, row['Name'], row['Company'], extract_year_month(row['Payment Cycle Period (Calculated)']), nightmare_set) and extract_year_month(row['Payment Cycle Period (Calculated)']) == target_month:
                print(f"Revenue row used at line {i+2} (including header):")
                print(row)
                found = True
                # Find all matching cost rows above
                print("\nCost rows matched and subtracted:")
                for j, crow in enumerate(reader[:i]):
                    if not is_revenue_row(crow):
                        if crow['Company'] and crow['Payment Cycle Period (Calculated)']:
                            if any_word_match(row['Name'], crow['Name'], crow['Company'], extract_year_month(crow['Payment Cycle Period (Calculated)']), nightmare_set) and \
                               normalize(crow['Company']) == normalize(row['Company']) and \
                               extract_year_month(crow['Payment Cycle Period (Calculated)']) == target_month:
                                print(f"  Line {j+2}: {crow}")
                break
    if not found:
        print("No revenue row found for Roos van der Pluijm in June 2025.")

def debug_roos_june2025_detailed():
    target_name = 'Roos van der Pluijm'
    target_month = '2025-06'
    nightmare_set = load_nightmare_matches()
    with open(FILENAME, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    found = False
    for i, row in enumerate(reader):
        if is_revenue_row(row):
            # Check name and month match
            rev_norm_company = normalize(row['Company'])
            rev_norm_month = extract_year_month(row['Payment Cycle Period (Calculated)'])
            rev_norm_name = normalize(row['Name'])
            if any_word_match(target_name, row['Name'], row['Company'], rev_norm_month, nightmare_set) and rev_norm_month == target_month:
                print(f"Revenue row at line {i+2} (including header):")
                print(row)
                print(f"Normalized revenue: company='{rev_norm_company}', month='{rev_norm_month}', name='{rev_norm_name}'")
                found = True
                print("\nCost rows matched and subtracted:")
                for j, crow in enumerate(reader[:i]):
                    if not is_revenue_row(crow):
                        cost_norm_company = normalize(crow['Company'])
                        cost_norm_month = extract_year_month(crow['Payment Cycle Period (Calculated)'])
                        cost_norm_name = normalize(crow['Name'])
                        match = False
                        if crow['Company'] and crow['Payment Cycle Period (Calculated)']:
                            if any_word_match(row['Name'], crow['Name'], crow['Company'], cost_norm_month, nightmare_set) and \
                               cost_norm_company == rev_norm_company and \
                               cost_norm_month == rev_norm_month:
                                match = True
                        print(f"  Line {j+2}: {crow}")
                        print(f"    Normalized cost: company='{cost_norm_company}', month='{cost_norm_month}', name='{cost_norm_name}', MATCH={match}")
                break
    if not found:
        print("No revenue row found for Roos van der Pluijm in June 2025.")

def debug_roos_pair_match():
    rev_name = 'Roos van der Pluijm'
    cost_name = 'Roos Juliette Van Der Pluijm'
    target_month = '2025-06'
    nightmare_set = load_nightmare_matches()
    with open(FILENAME, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    rev_row = None
    cost_row = None
    for row in rows:
        if is_revenue_row(row) and normalize(row['Name']) == normalize(rev_name) and extract_year_month(row['Payment Cycle Period (Calculated)']) == target_month:
            rev_row = row
        if not is_revenue_row(row) and normalize(row['Name']) == normalize(cost_name) and extract_year_month(row['Payment Cycle Period (Calculated)']) == target_month:
            cost_row = row
    if not rev_row or not cost_row:
        print('Could not find both revenue and cost rows.')
        return
    rev_norm_company = normalize(rev_row['Company'])
    rev_norm_month = extract_year_month(rev_row['Payment Cycle Period (Calculated)'])
    rev_norm_name = normalize(rev_row['Name'])
    cost_norm_company = normalize(cost_row['Company'])
    cost_norm_month = extract_year_month(cost_row['Payment Cycle Period (Calculated)'])
    cost_norm_name = normalize(cost_row['Name'])
    print('Revenue row:')
    print(rev_row)
    print(f"Normalized: name='{rev_norm_name}', company='{rev_norm_company}', month='{rev_norm_month}'")
    print('Cost row:')
    print(cost_row)
    print(f"Normalized: name='{cost_norm_name}', company='{cost_norm_company}', month='{cost_norm_month}'")
    # Check nightmare match
    blocked = False
    n1 = rev_norm_name
    n2 = cost_norm_name
    company = rev_norm_company
    month = rev_norm_month
    if ((n1, n2, company, month) in nightmare_set or
        (n2, n1, company, month) in nightmare_set):
        blocked = True
    print(f"Nightmare match blocks this pair: {blocked}")
    # Check name word match
    ignore_words = {'mark'}
    words1 = [w for w in re.findall(r'\w+', rev_name.lower()) if w not in ignore_words]
    words2 = [w for w in re.findall(r'\w+', cost_name.lower()) if w not in ignore_words]
    word_match = any(w1 == w2 for w1 in words1 for w2 in words2)
    print(f"Any word match: {word_match}")
    # Final match logic
    company_match = rev_norm_company == cost_norm_company
    month_match = rev_norm_month == cost_norm_month
    print(f"Company match: {company_match}, Month match: {month_match}")
    final_match = company_match and month_match and word_match and not blocked
    print(f"Final match: {final_match}")

def debug_roos_june2025_profit():
    target_name = 'Roos van der Pluijm'
    target_month = '2025-06'
    nightmare_set = load_nightmare_matches()
    with open(FILENAME, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    rev_row = None
    for i, row in enumerate(rows):
        if is_revenue_row(row) and normalize(row['Name']) == normalize(target_name) and extract_year_month(row['Payment Cycle Period (Calculated)']) == target_month:
            rev_row = (i, row)
            break
    if not rev_row:
        print('Revenue row not found.')
        return
    idx, row = rev_row
    payment = parse_payment_amount(row.get('Payment Amount (payment currency)', ''))
    print(f"Revenue amount: {payment}")
    cost_sum = 0.0
    print("Matched cost rows:")
    for j in range(idx):
        cost_row = rows[j]
        if not is_revenue_row(cost_row) and loose_match(row, cost_row, nightmare_set):
            cost = parse_payment_amount(cost_row.get('Payment Amount (payment currency)', ''))
            cost_sum += cost
            print(f"  Line {j+2}: {cost_row['Name']} | Amount: {cost}")
    print(f"Total cost sum: {cost_sum}")
    print(f"Final profit: {payment - cost_sum}")

def debug_large_negative_profits():
    with open(FILENAME, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print("Profit rows with payment amount < -1000:")
    for i, row in enumerate(rows):
        if row.get('Bucket', '').strip().lower() == 'profit':
            val = row.get('Payment Amount (payment currency)', '').replace(',', '').replace('€', '').strip()
            try:
                amount = float(val)
            except Exception:
                continue
            if amount < -1000:
                print(f"Row {i+2}: {row}")

def debug_aktaran_sevim_oct2022():
    target_name = 'Aktaran Sevim'
    target_month = '2022-10'
    nightmare_set = load_nightmare_matches()
    with open(FILENAME, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    rev_row = None
    for i, row in enumerate(rows):
        if is_revenue_row(row) and normalize(row['Name']) == normalize(target_name) and extract_year_month(row['Payment Cycle Period (Calculated)']) == target_month:
            rev_row = (i, row)
            break
    if not rev_row:
        print('Revenue row not found.')
        return
    idx, row = rev_row
    payment = parse_payment_amount(row.get('Payment Amount (payment currency)', ''))
    print(f"Revenue row (line {idx+2}): {row}")
    print(f"Revenue amount: {payment}")
    cost_sum = 0.0
    print("Matched cost rows:")
    for j in range(idx):
        cost_row = rows[j]
        if not is_revenue_row(cost_row) and loose_match(row, cost_row, nightmare_set):
            cost = parse_payment_amount(cost_row.get('Payment Amount (payment currency)', ''))
            cost_sum += cost
            print(f"  Line {j+2}: {cost_row}")
    print(f"Total cost sum: {cost_sum}")
    print(f"Final profit: {payment - cost_sum}")

if __name__ == '__main__':
    if '--roos-june2025-verbose' in sys.argv:
        debug_roos_june2025_verbose()
    elif '--roos-june2025-debug' in sys.argv:
        debug_roos_june2025()
    elif '--nightmare-matches' in sys.argv:
        find_nightmare_matches()
    elif '--mark-jacobs-may2024-debug' in sys.argv:
        debug_mark_jacobs_may2024()
    elif '--shareen-may2025-debug' in sys.argv:
        print_shareen_may2025_debug()
    elif '--roos-june2025-trace' in sys.argv:
        trace_roos_june2025()
    elif '--roos-june2025-detailed' in sys.argv:
        debug_roos_june2025_detailed()
    elif '--roos-pair-debug' in sys.argv:
        debug_roos_pair_match()
    elif '--roos-june2025-profit-debug' in sys.argv:
        debug_roos_june2025_profit()
    elif '--large-negative-profits' in sys.argv:
        debug_large_negative_profits()
    elif '--aktaran-sevim-oct2022-debug' in sys.argv:
        debug_aktaran_sevim_oct2022()
        sys.exit(0)
    else:
        main() 