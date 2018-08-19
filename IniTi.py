# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 16:41:16 2018

@author: hsasa
"""
# importing intitial libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Modules import dataCleaning as dc
from Modules import classs as cp
import seaborn as sns

#importing datasets

data_Matches = pd.read_csv("C:/Users/hsasa/Desktop/project1/DataSets/WorldCupMatches (1).csv")
data_Players = pd.read_csv("C:/Users/hsasa/Desktop/project1/DataSets/WorldCupPlayers (1).csv")
data_Cups    = pd.read_csv("C:/Users/hsasa/Desktop/project1/DataSets/WorldCups (1).csv")


#2018 player data

data_2018 = []

data_2018.extend([{
    'country': 'russia',
    'group': 'A',
    'coach': 'Cherchesov Stanislav',
    'name': 'Igor Akinfeev, Vladimir Gabulov, Andrey Lunev; Sergei Ignashevich, Mario Fernandes, Vladimir Granat, Fyodor Kudryashov, Andrei Semyonov, Igor Smolnikov, Ilya Kutepov, Aleksandr Yerokhin, Yuri Zhirkov, Daler Kuzyaev, Aleksandr Golovin, Alan Dzagoev, Roman Zobnin, Aleksandr Samedov, Yuri Gazinsky, Anton Miranchuk, Denis Cheryshev, Artyom Dzyuba, Aleksei Miranchuk, Fyodor Smolov'
}, {
    'country': 'saudi_arabia',
    'group': 'A',
    'coach': 'Pizzi Juan Antonio',
    'name': 'Mohammed Al-Owais, Yasser Al-Musailem, Abdullah Al-Mayuf; Mansoor Al-Harbi, Yasser Al-Shahrani, Mohammed Al-Burayk, Motaz Hawsawi, Osama Hawsawi, Ali Al-Bulaihi, Omar Othman; Abdullah Alkhaibari, Abdulmalek Alkhaibri, Abdullah Otayf, Taiseer Al-Jassam, Hussain Al-Moqahwi, Salman Al-Faraj, Mohamed Kanno, Hatan Bahbir, Salem Al-Dawsari, Yahia Al-Shehri, Fahad Al-Muwallad, Mohammad Al-Sahlawi, Muhannad Assiri'
}, {
    'country': 'egypt',
    'group': 'A',
    'coach': 'Cuper Hector',
    'name': 'Essam El Hadary, Mohamed El-Shennawy, Sherif Ekramy; Ahmed Fathi, Abdallah Said, Saad Samir, Ayman Ashraf, Mohamed Abdel-Shafy, Ahmed Hegazi, Ali Gabr, Ahmed Elmohamady, Omar Gaber; Tarek Hamed, Mahmoud Shikabala, Sam Morsy, Mohamed Elneny, Mahmoud Kahraba, Ramadan Sobhi, Trezeguet, Amr Warda; Marwan Mohsen, Mohamed Salah, Mahmoud Elwensh'
}, {
    'country': 'uruguay',
    'group': 'A',
    'coach': 'Tabarez Oscar',
    'name': 'Fernando Muslera, Martin Silva, Martin Campana, Diego Godin, Sebastian Coates, Jose Maria Gimenez, Maximiliano Pereira, Gaston Silva, Martin Caceres, Guillermo Varela, Nahitan Nandez, Lucas Torreira, Matias Vecino, Rodrigo Bentancur, Carlos Sanchez, Giorgian De Arrascaeta, Diego Laxalt, Cristian Rodriguez, Jonathan Urretaviscaya, Cristhian Stuani, Maximiliano Gomez, Edinson Cavani, Luis Suarez'
}, {
    'country': 'portugal',
    'group': 'B',
    'coach': 'Santos Fernando',
    'name': 'Anthony Lopes, Beto, Rui Patricio, Bruno Alves, Cedric Soares, Jose Fonte, Mario Rui, Pepe, Raphael Guerreiro, Ricardo Pereira, Ruben Dias, Adrien Silva, Bruno Fernandes, Joao Mario, Joao Moutinho, Manuel Fernandes, William Carvalho, Andre Silva, Bernardo Silva, Cristiano Ronaldo, Gelson Martins, Goncalo Guedes, Ricardo Quaresma'
}, {
    'country': 'spain',
    'group': 'B',
    'coach': 'Hierro Fernando',
    'name': 'David de Gea, Pepe Reina, Kepa Arrizabalaga; Dani Carvajal, Alvaro Odriozola, Gerard Pique, Sergio Ramos, Nacho, Cesar Azpilicueta, Jordi Alba, Nacho Monreal; Sergio Busquets, Saul Niquez, Koke, Thiago Alcantara, Andres Iniesta, David Silva; Isco, Marcio Asensio, Lucas Vazquez, Iago Aspas, Rodrigo, Diego Costa'
}, {
    'country': 'morocco',
    'group': 'B',
    'coach': 'Renard Herve',
    'name': "Mounir El Kajoui, Yassine Bounou, Ahmad Reda Tagnaouti, Mehdi Benatia, Romain Saiss, Manuel Da Costa, Badr Benoun, Nabil Dirar, Achraf Hakimi, Hamza Mendyl; M'bark Boussoufa, Karim El Ahmadi, Youssef Ait Bennasser, Sofyan Amrabat, Younes Belhanda, Faycal Fajr, Amine Harit; Khalid Boutaib, Aziz Bouhaddouz, Ayoub El Kaabi, Nordin Amrabat, Mehdi Carcela, Hakim Ziyech"
}, {
    'country': 'iran',
    'group': 'B',
    'coach': 'Queiroz Carlos',
    'name': 'Alireza Beiranvand, Rashid Mazaheri, Amir Abedzadeh; Ramin Rezaeian, Mohammad Reza Khanzadeh, Morteza Pouraliganji, Pejman Montazeri, Seyed Majid Hosseini, Milad Mohammadi, Roozbeh Cheshmi; Saeid Ezatolahi, Masoud Shojaei, Saman Ghoddos, Mehdi Torabi, Ashkan Dejagah, Omid Ebrahimi, Ehsan Hajsafi, Vahid Amiri; Alireza Jahanbakhsh, Karim Ansarifard, Mahdi Taremi, Sardar Azmoun, Reza Ghoochannejhad'
}, {
    'country': 'france',
    'group': 'C',
    'coach': 'Deschamps Didier',
    'name': "Alphonse Areola, Hugo Lloris, Steve Mandanda; Lucas Hernandez, Presnel Kimpembe, Benjamin Mendy, Benjamin Pavard, Adil Rami, Djibril Sidibe, Samuel Umtiti, Raphael Varane; N'Golo Kante, Blaise Matuidi, Steven N'Zonzi, Paul Pogba, Corentin Tolisso, Ousmane Dembele, Nabil Fekir; Olivier Giroud, Antoine Griezmann, Thomas Lemar, Kylian Mbappe, Florian Thauvin"
}, {
    'country': 'australia',
    'group': 'C',
    'coach': 'Van Marwur bert',
    'name': 'Brad Jones, Mat Ryan, Danny Vukovic; Aziz Behich, Milos Degenek, Matthew Jurman, James Meredith, Josh Risdon, Trent Sainsbury; Jackson Irvine, Mile Jedinak, Robbie Kruse, Massimo Luongo, Mark Milligan, Aaron Mooy, Tom Rogic; Daniel Arzani, Tim Cahill, Tomi Juric, Mathew Leckie, Andrew Nabbout, Dimitri Petratos, Jamie Maclaren'
}, {
    'country': 'peru',
    'group': 'C',
    'coach': 'Gareca Ricardo',
    'name': 'Carlos Caceda, Jose Carvallo, Pedro Gallese, Luis Advincula, Pedro Aquino, Miguel Araujo, Andre Carrillo, Wilder Cartagena, Aldo Corzo, Christian Cueva, Jefferson Farfan, Edison Flores, Paolo Hurtado, Nilson Loyola, Andy Polo, Christian Ramos, Alberto Rodriguez, Raul Ruidiaz, Anderson Santamaria, Renato Tapia, Miguel Trauco, Yoshimar Yotun, Paolo Guerrero'
}, {
    'country': 'denmark',
    'group': 'C',
    'coach': 'Hareide Age',
    'name': 'Kasper Schmeichel, Jonas Lossl, Frederik Ronow; Simon Kjaer, Andreas Christensen, Mathias Jorgensen, Jannik Vestergaard, Henrik Dalsgaard, Jens Stryger, Jonas Knudsen; William Kvist, Thomas Delaney, Lukas Lerager, Lasse Schone, Christian Eriksen, Michael Krohn-Dehli; Pione Sisto, Martin Braithwaite, Andreas Cornelius, Viktor Fischer, Yussuf Poulsen, Nicolai Jorgensen, Kasper Dolberg'
}, {
    'country': 'argentina',
    'group': 'D',
    'coach': 'Sampaoli Jorge',
    'name': 'Nahuel Guzmán, Willy Caballero, Franco Armani; Gabriel Mercado, Nicolas Otamendi, Federico Fazio, Nicolas Tagliafico, Marcos Rojo, Marcos Acuna, Cristian Ansaldi, Eduardo Salvio; Javier Mascherano, Angel Di Maria, Ever Banega, Lucas Biglia, Manuel Lanzini, Gio Lo Celso, Maximiliano Meza; Lionel Messi, Sergio Aguero, Gonzalo Higuain, Paulo Dybala, Cristian Pavon'
}, {
    'country': 'iceland',
    'group': 'D',
    'coach': 'Hallgrimsson Heimir',
    'name': 'Hannes Thor Halldorsson, Runar Alex Runarsson, Frederik Schram; Kari Arnason, Ari Freyr Skulason, Birkir Mar Saevarsson, Sverrir Ingi Ingason, Hordur Magnusson, Holmar Orn Eyjolfsson, Ragnar Sigurdsson; Johann Berg Gudmundsson, Birkir Bjarnason, Arnor Ingvi Traustason, Emil Hallfredsson, Gylfi Sigurdsson, Olafur Ingi Skulason, Rurik Gislason, Samuel Fridjonsson, Aron Gunnarsson; Alfred Finnbogason, Bjorn Bergmann Sigurdarson, Jon Dadi Bodvarsson, Albert Gudmundsson'
}, {
    'country': 'croatia',
    'group': 'D',
    'coach': 'Dalic Zlatko',
    'name': 'Danijel Subasic, Lovre Kalinic, Dominik Livakovic; Vedran Corluka, Domagoj Vida, Ivan Strinic, Dejan Lovren, Sime Vrsaljko, Josip Pivaric, Tin Jedvaj, Duje Caleta-Car; Luka Modric, Ivan Rakitic, Mateo Kovacic, Milan Badelj, Marcelo Brozovic, Filip Bradaric; Mario Mandzukic, Ivan Perisic, Nikola Kalinic, Andrej Kramaric, Marko Pjaca, Ante Rebic'
}, {
    'country': 'nigeria',
    'group': 'D',
    'coach': 'Rohr Gernot',
    'name': 'Ikechukwu Ezenwa, Daniel Akpeyi, Francis Uzoho; William Troost-Ekong, Leon Balogun, Kenneth Omeruo, Bryan Idowu, Chidozie Awaziem, Abdullahi Shehu, Elderson Echiejile, Tyronne Ebuehi; John Obi Mikel, Ogenyi Onazi, John Ogu, Wilfred Ndidi, Oghenekaro Etebo, Joel Obi; Odion Ighalo, Ahmed Musa, Victor Moses, Alex Iwobi, Kelechi Iheanacho, Simeon Nwankwo'
}, {
    'country': 'brazil',
    'group': 'E',
    'coach': 'Tite',
    'name': ' Alisson, Ederson, Cassio; Danilo, Fagner, Marcelo, Filipe Luis, Thiago Silva, Marquinhos, Miranda, Pedro Geromel; Casemiro, Fernandinho, Paulinho, Fred, Renato Augusto, Philippe Coutinho, Willian, Douglas Costa; Neymar, Taison, Gabriel Jesus, Roberto Firmino'
}, {
    'country': 'switzerland',
    'group': 'E',
    'coach': 'Petkovic Vladimir',
    'name': 'Roman Burki, Yvon Mvogo, Yann Sommer; Manuel Akanji, Johan Djourou, Nico Elvedi, Michael Lang, Stephan Lichtsteiner, Jacques-Francois Moubandje, Ricardo Rodriguez, Fabian Schaer; Valon Behrami, Blerim Dzemaili, Gelson Fernandes, Remo Freuler, Xherdan Shaqiri, Granit Xhaka, Steven Zuber, Denis Zakaria; Josip Drmic, Breel Embolo, Mario Gavranovic, Haris Seferovic'
}, {
    'country': 'costa_rica',
    'group': 'E',
    'coach': 'Ranurez Oscar',
    'name': 'Keylor Navas, Patrick Pemberton, Leonel Moreira, Cristian Gamboa, Ian Smith, Ronald Matarrita, Bryan Oviedo, Oscar Duarte, Giancarlo Gonzalez, Francisco Calvo, Kendall Waston, Johnny Acosta, David Guzman, Yeltsin Tejeda, Celso Borges, Randall Azofeifa, Rodney Wallace, Bryan Ruiz, Daniel Colindres, Christian Bolanos, Johan Venegas, Joel Campbell, Marco Urena'
}, {
    'country': 'serbia',
    'group': 'E',
    'coach': 'Krstajic Mladen',
    'name': ' Vladimir Stojkovic, Predrag Rajkovic, Marko Dmitrovic, Aleksandar Kolarov, Antonio Rukavina, Milan Rodic, Branislav Ivanovic, Uros Spajic, Milos Veljkovic, Dusko Tosic, Nikola Milenkovic; Nemanja Matic, Luka Milivojevic, Marko Grujic, Dusan Tadic, Andrija Zivkovic, Filip Kostic, Nemanja Radonjic, Sergej Milinkovic-Savic, Adem Ljajic; Aleksandar Mitrovic, Aleksandar Prijovic, Luka Jovic'
}, {
    'country': 'germany',
    'group': 'F',
    'coach': 'Low Joachim',
    'name': 'Manuel Neuer, Marc-Andre ter Stegen, Kevin Trapp; Jerome Boateng, Matthias Ginter, Jonas Hector, Mats Hummels, Joshua Kimmich, Marvin Plattenhardt, Antonio Rudiger, Niklas Sule; Julian Brandt, Julian Draxler, Mario Gomez, Leon Goretzka, Ilkay Gundogan, Sami Khedira, Toni Kroos, Thomas Muller, Mesut Ozil, Marco Reus, Sebastian Rudy, Timo Werner'
}, {
    'country': 'mexico',
    'group': 'F',
    'coach': 'Osorio Juan Carlos',
    'name': 'Jesus Corona, Alfredo Talavera, Guillermo Ochoa; Hugo Ayala, Carlos Salcedo, Diego Reyes, Miguel Layun, Hector Moreno, Edson Alvarez; Rafael Marquez, Jonathan dos Santos, Marco Fabian, Giovani dos Santos, Hector Herrera, Andres Guardado; Raul Jimenez, Carlos Vela, Javier Hernandez, Jesus Corona, Oribe Peralta, Javier Aquino, Hirving Lozano'
}, {
    'country': 'sweden',
    'group': 'F',
    'coach': 'Andersson Janne',
    'name': 'Robin Olsen, Karl-Johan Johnsson, Kristoffer Nordfeldt, Mikael Lustig, Victor Lindelof, Andreas Granqvist, Martin Olsson, Ludwig Augustinsson, Filip Helander, Emil Krafth, Pontus Jansson, Sebastian Larsson, Albin Ekdal, Emil Forsberg, Gustav Svensson, Oscar Hiljemark, Viktor Claesson, Marcus Rohden, Jimmy Durmaz, Marcus Berg, John Guidetti, Ola Toivonen, Isaac Kiese Thelin'
}, {
    'country': 'south_korea',
    'group': 'F',
    'coach': 'Shin Taeyong',
    'name': 'Kim Seunggyu, Kim Jinhyeon, Cho Hyeonwoo, Kim Younggwon, Jang Hyunsoo, Jeong Seunghyeon, Yun Yeongseon, Oh Bansuk, Kim Minwoo, Park Jooho, Hong Chul, Go Yohan, Lee Yong, Ki Sungyueng, Jeong Wooyoung, Ju Sejong, Koo Jacheol, Lee Jaesung, Lee Seungwoo, Moon Sunmin, Kim Shinwook, Son Heungmin, Hwang Heechan'
}, {
    'country': 'belgium',
    'group': 'G',
    'coach': 'Martinez Roberto',
    'name': 'Koen Casteels, Thibaut Courtois, Simon Mignolet; Toby Alderweireld, Dedryck Boyata, Vincent Kompany, Thomas Meunier, Thomas Vermaelen, Jan Vertonghen; Nacer Chadli, Kevin De Bruyne, Mousa Dembele, Leander Dendoncker, Marouane Fellaini, Youri Tielemans, Axel Witsel; Michy Batshuayi, Yannick Carrasco, Eden Hazard, Thorgan Hazard, Adnan Januzaj, Romelu Lukaku, Dries Mertens'
}, {
    'country': 'panama',
    'group': 'G',
    'coach': 'Gomez Hernan',
    'name': 'Jose Calderon, Jaime Penedo, Alex Rodríguez; Felipe Baloy, Harold Cummings, Eric Davis, Fidel Escobar, Adolfo Machado, Michael Murillo, Luis Ovalle, Roman Torres; Edgar Barcenas, Armando Cooper, Anibal Godoy, Gabriel Gomez, Valentin Pimentel, Alberto Quintero, Jose Luis Rodriguez; Abdiel Arroyo, Ismael Diaz, Blas Perez, Luis Tejada, Gabriel Torres'
}, {
    'country': 'tunisia',
    'group': 'G',
    'coach': 'Maaloul Nabil',
    'name': 'Farouk Ben Mustapha, Moez Hassen, Aymen Mathlouthi, Rami Bedoui, Yohan Benalouane, Syam Ben Youssef, Dylan Bronn, Oussama Haddadi, Ali Maaloul, Yassine Meriah, Hamdi Nagguez, Anice Badri, Mohamed Amine Ben Amor, Ghaylene Chaalali, Ahmed Khalil, Saifeddine Khaoui, Ferjani Sassi, Ellyes Skhiri, Naim Sliti, Bassem Srarfi, Fakhreddine Ben Youssef, Saber Khalifa, Wahbi Khazri'
}, {
    'country': 'england',
    'group': 'G',
    'coach': 'Southgate Gareth',
    'name': 'Jack Butland, Nick Pope, Jordan Pickford; Fabian Delph, Danny Rose, Eric Dier, Kyle Walker, Kieran Trippier, Trent Alexander-Arnold, Harry Maguire, John Stones, Phil Jones, Gary Cahill; Jordan Henderson, Jesse Lingard, Ruben Loftus-Cheek, Ashley Young, Dele Alli, Raheem Sterling; Harry Kane, Jamie Vardy, Marcus Rashford, Danny Welbeck'
}, {
    'country': 'poland',
    'group': 'H',
    'coach': 'Nawalka Adam',
    'name': 'Bartosz Bialkowski, Lukasz Fabianski, Wojciech Szczesny; Jan Bednarek, Bartosz Bereszynski, Thiago Cionek, Kamil Glik, Artur Jedrzejczyk, Michal Pazdan, Lukasz Piszczek; Jakub Blaszczykowski, Jacek Goralski, Kamil Goricki, Grzegorz Krychowiak, Slawomir Peszko, Maciej Rybus, Piotr Zielinski, Rafal Kurzawa, Karol Linetty; Dawid Kownacki, Robert Lewandowski, Arkadiusz Milik, Lukasz Teodorczyk'
}, {
    'country': 'senegal',
    'group': 'H',
    'coach': 'Cisse Aliou',
    'name': 'Abdoulaye Diallo, Khadim Ndiaye, Alfred Gomis, Lamine Gassama, Moussa Wague, Saliou Ciss, Youssouf Sabaly, Kalidou Koulibaly, Salif Sane, Cheikhou Kouyate, Kara Mbodji, Idrisa Gana Gueye, Cheikh Ndoye, Alfred Ndiaye, Pape Alioune Ndiaye, Moussa Sow, Moussa Konate, Diafra Sakho, Sadio Mane, Ismaila Sarr, Mame Biram Diouf, Mbaye Niang, Diao Keita Balde'
}, {
    'country': 'colombia',
    'group': 'H',
    'coach': 'Pekerman Jose',
    'name': 'David Ospina, Camilo Vargas, Jose Fernando Cuadrado; Cristian Zapata, Davinson Sanchez, Santiago Arias, Oscar Murillo, Frank Fabra, Johan Mojica, Yerry Mina; Wilmar Barrios, Carlos Sanchez, Jefferson Lerma, Jose Izquierdo, James Rodriguez, Abel Aguilar, Juan Fernando Quintero, Mateus Uribe, Juan Guillermo Cuadrado; Radamel Falcao Garcia, Miguel Borja, Carlos Bacca, Luis Fernando Muriel'
}, {
    'country': 'japan',
    'group': 'H',
    'coach': 'Nishino Akira',
    'name': 'Eiji Kawashima, Masaaki Higashiguchi, Kosuke Nakamura, Yuto Nagatomo, Tomoaki Makino, Maya Yoshida, Hiroki Sakai, Gotoku Sakai, Gen Shoji, Wataru Endo, Naomichi Ueda, Makoto Hasebe, Keisuke Honda, Takashi Inui, Shinji Kagawa, Hotaru Yamaguchi, Genki Haraguchi, Takashi Usami, Gaku Shibasaki, Ryota Oshima, Shinji Okazaki, Yuya Osako, Yoshinori Muto'
}])

data_2018 = pd.DataFrame(data_2018)

#data_2018.to_csv("data2018.csv")


#Data cleaning
#1st clearing data_2018
data_2018['participants'] = data_2018.apply(dc.clean_merge, axis=1)
data_2018['label'] = 0

#2nd getting codes to name and names to code with position

code_name, name_code, name_pos, pos_name = dc.country_names(data_Matches) 

data_Matches['home_participants'] = data_Matches.apply(lambda x: dc.partic(data_Players, x['MatchID'], x['Home Team Initials']), axis = 1)
data_Matches['away_participants'] = data_Matches.apply(lambda x: dc.partic(data_Players, x['MatchID'], x['Away Team Initials']), axis = 1)


data_train =[]
winner_name = []

for i,row in data_Cups.iterrows():
    data_train.append({'label': 1,'name': dc.build_rec(data_Matches, row['Year'],row['Winner'])})
    data_train.append({'label': 2,'name': dc.build_rec(data_Matches, row['Year'],row['Runners-Up'])})
    data_train.append({'label': 3,'name': dc.build_rec(data_Matches, row['Year'],row['Third'])})
    data_train.append({'label': 4,'name': dc.build_rec(data_Matches, row['Year'],row['Fourth'])})

    winner_name =([row['Winner'], row['Runners-Up'], row['Third'], row['Fourth']])
    
    results = dc.Neg_Rec(data_Matches,row['Year'], winner_name)
    data_train.extend(results)
    
data_training = pd.DataFrame(data_train)    
dt = data_training
#dt['1'], dt['2'], dt['3'], dt['4'], dt['5'], dt['6'], dt['7'], dt['8'], dt['9'], dt['10'], dt['11'], dt['12'], dt['13'], dt['14'], dt['15'], dt['16'] = zip(*dt['name'].map(lambda x:  x.split('  ')))
#dt.drop(['name'], axis = 1, inplace = True)

from sklearn.model_selection import train_test_split as ts
train_df, test_df = ts(dt , test_size = 0.01) 

char_cnn = cp.CharCNN(max_len_s=256, max_num_s=1)
char_cnn.preporcess(labels=dt['label'].unique())

x_train,y_train = char_cnn.process(df = train_df,x_col='name',y_col='label')
x_test,y_test = char_cnn.process(df = test_df,x_col='name',y_col='label')

char_cnn.build_model()
char_cnn.train(x_train, y_train, x_test, y_test, batch_size = 32, epochs = 10)

y_pred = char_cnn.predict(x_test)

plt.figure(figsize=(12,6))
sns.countplot(data_Cups['Winner'])

x_2018, y_2018 = char_cnn.process(data_2018, x_col = 'name', y_col = 'label')
y_pred = char_cnn.predict(x_2018)

a=[]
a = dc.predictname(y_pred)
from sklearn.metrics import accuracy_score as cm
c = cm(y_test,y_pred)*100
























