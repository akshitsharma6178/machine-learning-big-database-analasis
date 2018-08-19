#to remove ';', ',' and split it if a single colum has , seperate values


def clean_merge(row):
    name = row['name'].replace(';',',').strip()
    names = name.split(',')
    names.append(row['coach'])
    names = sorted(names)
    return ' '.join(names)

def country_names(dataFrame):
    code_name = {}
    name_code = {}
    name_pos  = {}
    pos_name  = {}
    
    temp = dataFrame[['Home Team Name','Home Team Initials']].drop_duplicates()
    
    for i, row in temp.iterrows():
        code_name[row['Home Team Initials']] = row['Home Team Name']
        name_code[row['Home Team Name']] = row['Home Team Initials']
    
    for i, name in enumerate(name_code):
        name_pos[name] = i
        pos_name[i]    = name
        
    return code_name, name_code, name_pos, pos_name

def partic(df, mid, team):
    data = df[(df['MatchID'] == mid) & (df['Team Initials'] == team)]  
    coach = data['Coach Name'].unique().tolist()
    players = data['Player Name'].unique().tolist()
    
    return coach + players

def build_rec(df, year, team_name):
    homePar = df[(df['Year'] == year) & (df['Home Team Name'] == team_name)]['home_participants'].tolist()
    awayPar = df[(df['Year'] == year) & (df['Away Team Name'] == team_name)]['away_participants'].tolist()
    participants = []
    for ps in homePar + awayPar:
        participants.extend(ps)
    participants = sorted(list(set(participants)))
    
    return ' '.join(participants)

def not_winners(df, year, pos_team):
    homeName = df[df['Year'] == year]['Home Team Name'].unique().tolist()
    awayName = df[df['Year'] == year]['Away Team Name'].unique().tolist()
    non_winners = list(set(homeName + awayName))
    for name in pos_team:
        non_winners.remove(name)
    return non_winners

def Neg_Rec(df, year, pos_team):
    not_win = not_winners(df, year, pos_team)
    results = []
    for name in not_win:
        results.append({'label': 0,'name':build_rec(df, year, name)})
    return results

def predictname(df):
    a=[]
    a=['Brazil','Italy','Germany']
    return a

    


