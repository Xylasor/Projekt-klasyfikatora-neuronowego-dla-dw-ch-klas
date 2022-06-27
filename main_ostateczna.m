clc;           
clear;      
close all
% Import danych z pliku tekstowego
dane_ucz=importdata('data_train.txt');
dane_test=importdata('data_test.txt');
% Opis tablicy 'dane':
% kolumny 1,2 - współrzędne punktów do klasyfikacji
% kolumna 3   - etykieta punktu {-1,1}

prog_dolny = 0.45;
prog_gorny = 0.55;
N = 20;             % Liczba iteracji do usrednienia
liczba_neuronow_ukrytych=9;

wart_kl_poz=1;                      %wartości etykiet używane przy uczeniu
wart_kl_neg=0;
idx_poz=find(dane_ucz(:,3)==1);     %indeksy punktów klasy pozytywnej
idx_neg=find(dane_ucz(:,3)~=1);     %indeksy punktów klasy negatywnej
dane_ucz(idx_poz,3)=wart_kl_poz;    %zmiana wartości etykiet klasy poz.
dane_ucz(idx_neg,3)=wart_kl_neg;    %zmiana wartości etykiet klasy neg.

best_proc_1 = 0;
suma_proc_dobrze_1 = 0;
suma_proc_dokl_1 = 0;
suma_proc_prec_1 = 0;
suma_proc_niekl_1 = 0;
suma_proc_niepop_1 = 0;
for j = 1:N
    [net]=train_net(dane_ucz(:,1:2),dane_ucz(:,3),liczba_neuronow_ukrytych);
    for i=1:size(dane_ucz,1)
        y(i) = net(dane_ucz(i,[1,2])');
    end
    
    ind_poz = find(y > prog_gorny);
    ind_neg = find(y < prog_dolny);
    ind_brak = find(y>prog_dolny & y<prog_gorny);
    suma_proc_niekl_1 = suma_proc_niekl_1 + numel(ind_brak)/size(dane_ucz,1)*100;
    suma_proc_niepop_1 = suma_proc_niepop_1 + (numel(find(ismember(ind_neg,idx_poz)))+numel(find(ismember(ind_poz,idx_neg))))/size(dane_ucz,1)*100;
    
    proc_dokl = numel(find(ismember(ind_poz,idx_poz)))/numel(idx_poz)*100;
    suma_proc_dokl_1 = suma_proc_dokl_1 + proc_dokl;
    proc_prec = numel(find(ismember(ind_neg,idx_neg)))/numel(idx_neg)*100;
    suma_proc_prec_1 = suma_proc_prec_1 + proc_prec;
    proc_dobrze = proc_dokl + proc_prec;
    suma_proc_dobrze_1 = suma_proc_dobrze_1 + proc_dobrze;
    
    %proc_dobrze = (numel(find(ismember(ind_poz,idx_poz)))+numel(find(ismember(ind_neg,idx_neg))))/size(dane_ucz,1)*100;
    if proc_dobrze > best_proc_1
        best_proc_1 = proc_dobrze;
        best_net_1 = net;
        w_1 = net.IW{1}';
        bin_1 = net.b{1};
        v_1 = net.LW{2,1}';
        bout_1 = net.b{2};
    end
end

wyniki_1 = [suma_proc_dokl_1, suma_proc_prec_1, suma_proc_dobrze_1, suma_proc_niepop_1, suma_proc_niekl_1]/N;


wart_kl_poz=1;                  %wartości etykiet używane przy uczeniu
wart_kl_neg=0;
idx_poz_2=find(dane_ucz(:,3)~=1);     %indeksy punktów klasy pozytywnej
idx_neg_2=find(dane_ucz(:,3)==1);     %indeksy punktów klasy negatywnej
dane_ucz(idx_poz_2,3)=wart_kl_poz;    %zmiana wartości etykiet klasy poz.
dane_ucz(idx_neg_2,3)=wart_kl_neg;    %zmiana wartości etykiet klasy neg.

best_proc_2 = 0;
suma_proc_dobrze_2 = 0;
suma_proc_dokl_2 = 0;
suma_proc_prec_2 = 0;
suma_proc_niekl_2 = 0;
suma_proc_niepop_2 = 0;
for j = 1:N
    [net]=train_net(dane_ucz(:,1:2),dane_ucz(:,3),liczba_neuronow_ukrytych);
    for i=1:size(dane_ucz,1)
        y(i) = net(dane_ucz(i,[1,2])');
    end
    
    ind_poz = find(y > prog_gorny);
    ind_neg = find(y < prog_dolny);
    ind_brak = find(y>prog_dolny & y<prog_gorny);
    suma_proc_niekl_2 = suma_proc_niekl_2 + numel(ind_brak)/size(dane_ucz,1)*100;
    suma_proc_niepop_2 = suma_proc_niepop_2 + (numel(find(ismember(ind_neg,idx_poz_2)))+numel(find(ismember(ind_poz,idx_neg_2))))/size(dane_ucz,1)*100;
    
    proc_prec = numel(find(ismember(ind_poz,idx_poz_2)))/numel(idx_poz_2)*100;
    suma_proc_prec_2 = suma_proc_prec_2 + proc_prec;
    proc_dokl = numel(find(ismember(ind_neg,idx_neg_2)))/numel(idx_neg_2)*100;
    suma_proc_dokl_2 = suma_proc_dokl_2 + proc_dokl;
    proc_dobrze = proc_dokl + proc_prec;
    suma_proc_dobrze_2 = suma_proc_dobrze_2 + proc_dobrze;
    
    if proc_dobrze > best_proc_2
        best_proc_2 = proc_dobrze;
        best_net_2 = net;
        w_2 = net.IW{1}';
        bin_2 = net.b{1};
        v_2 = net.LW{2,1}';
        bout_2 = net.b{2};
    end
end

wyniki_2 = [suma_proc_dokl_2, suma_proc_prec_2, suma_proc_dobrze_2, suma_proc_niepop_2, suma_proc_niekl_2]/N;


%% 

idx_poz=find(dane_test(:,3)==1);     %indeksy punktów klasy pozytywnej
idx_neg=find(dane_test(:,3)~=1);     %indeksy punktów klasy negatywnej
idx_poz_2=find(dane_test(:,3)~=1);     %indeksy punktów klasy pozytywnej
idx_neg_2=find(dane_test(:,3)==1);     %indeksy punktów klasy negatywnej

% Wynik klasyfikacji sieci na zbiorze uczacym i testowym
for i=1:size(dane_test,1)
    y_1(i) = best_net_1(dane_test(i,[1,2])');
    y_2(i) = best_net_2(dane_test(i,[1,2])');
end
ind_poz_1 = find(y_1 > prog_gorny);
ind_neg_1 = find(y_1 < prog_dolny);
ind_brak_1 = find(y_1>=prog_dolny & y_1<=prog_gorny);
TP = numel(find(ismember(ind_poz_1,idx_poz)));
FP = numel(find(ismember(ind_neg_1,idx_poz)));
proc_dobrze_1 = TP/numel(idx_poz)*100;
proc_zle_1 = FP/numel(idx_poz)*100;
proc_brak_1 = numel(find(ismember(ind_brak_1,idx_poz)))/numel(idx_poz)*100;

ind_poz_2 = find(y_2 > prog_gorny);
ind_neg_2 = find(y_2 < prog_dolny);
ind_brak_2 = find(y_2>=prog_dolny & y_2<=prog_gorny);
TN = numel(find(ismember(ind_poz_2,idx_poz_2)));
FN = numel(find(ismember(ind_neg_2,idx_poz_2)));
proc_dobrze_2 = TN/numel(idx_poz_2)*100;
proc_zle_2 = FN/numel(idx_poz_2)*100;
proc_brak_2 = numel(find(ismember(ind_brak_2,idx_poz_2)))/numel(idx_poz_2)*100;

wyniki_ucz = [proc_dobrze_1, TP, proc_zle_1, FP, proc_brak_1, numel(find(ismember(ind_brak_1,idx_poz)));...
    proc_dobrze_2, TN,proc_zle_2, FN,proc_brak_2, numel(find(ismember(ind_brak_2,idx_poz_2)))];


komp = TP/numel(idx_poz);
f_a = FP/numel(idx_neg);
prec = TP/(TP+FP);
dokl = (TP+TN)/(numel(idx_poz)+numel(idx_poz_2));
spec = TN/(FP+TN);

parametry_ucz =[komp; f_a; prec; dokl; spec]*100;

figure(1)
plot(dane_test(ind_poz_1(ismember(ind_poz_1,idx_poz)),1),dane_test(ind_poz_1(ismember(ind_poz_1,idx_poz)),2),'g+',...
    dane_test(ind_neg_1(ismember(ind_neg_1,idx_poz)),1),dane_test(ind_neg_1(ismember(ind_neg_1,idx_poz)),2),'r+',...
    dane_test(ind_poz_2(ismember(ind_poz_2,idx_poz_2)),1),dane_test(ind_poz_2(ismember(ind_poz_2,idx_poz_2)),2),'r*',...
    dane_test(ind_neg_2(ismember(ind_neg_2,idx_poz_2)),1),dane_test(ind_neg_2(ismember(ind_neg_2,idx_poz_2)),2),'g*')

hold on
for i=1:size(w_1,2)
    a = (-(bin_1(i)/w_1(2,i))/(bin_1(i)/w_1(1,i)));
    b = (-bin_1(i)/w_1(2,i));
    x_line = [-1 1];
    y_line = a*x_line+b;
    plot(x_line,y_line,'k')
    a = (-(bin_2(i)/w_2(2,i))/(bin_2(i)/w_2(1,i)));
    b = (-bin_2(i)/w_2(2,i));
    x_line = [-1 1];
    y_line = a*x_line+b;
    plot(x_line,y_line,'k')
end
hold off
xlim([-1 1])
ylim([-1 1])
title('Dane testowe')
xlabel('x_3')
ylabel('x_4')
legend('TP', 'FP', 'TN', 'FN')
