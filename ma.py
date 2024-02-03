import csv

# 球员名单
players_to_find = [
    "Carlos Alcaraz", "Alexander Zverev", "Frances Tiafoe",
    "Alejandro Davidovich Fokina", "Daniil Medvedev", "Jiri Lehecka",
    "Christopher Eubanks", "Laslo Djere", "Jannik Sinner",
    "Daniel Elahi Galan", "Guido Pella", "Denis Shapovalov",
    "Andrey Rublev", "Alexander Bublik", "Lorenzo Musetti",
    "Stan Wawrinka", "Grigor Dimitrov", "Roman Safiullin", "Hubert Hurkacz"
]

# 读取 CSV 文件并查找指定球员
filtered_players = []
with open('atp_players.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        full_name = f"{row['name_first']} {row['name_last']}"
        if full_name in players_to_find:
            filtered_players.append({
                'player_id': row['player_id'],
                'last_name': row['name_last'],
                'first_name': row['name_first'],
                'country': row['ioc']
            })

# 保存筛选后的信息到新的CSV文件
with open('filtered_players.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['player_id', 'last_name', 'first_name', 'country']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for player in filtered_players:
        writer.writerow(player)

# 207989
# Carlos	Alcaraz
# ESP
#
# 100644
# Alexander	Zverev
# GER
#
# 126207
# Frances	Tiafoe
# USA




