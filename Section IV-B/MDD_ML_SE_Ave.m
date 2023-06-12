function [] = MDD_ML_SE_Ave(SquareLength,Num_AP,Num_MS)

rng('shuffle')
%SquareLength = 100; %square 400m * 400m
%Num_AP = 24;
Num_AP_ant = 8;
%Num_MS = 6;
Num_DLsubcarrier = 4;
Num_ULsubcarrier = 2;
Num_Sumsubcarrier = Num_DLsubcarrier + Num_ULsubcarrier;
Power_AP_dBm = 40;
Power_AP_W = (10^(0.1 * Power_AP_dBm) * 10^-3);
Power_MS_dBm = 30;
Power_MS_W = (10^(0.1 * Power_MS_dBm) * 10^-3);
Power_Pi_dBm = 30;
Power_Pi_W = (10^(0.1 * Power_Pi_dBm) * 10^-3);
SI_cap_MS = 10^(-11);
SI_cap_AP = 10^(-12);
IMI_cap_MS = 10^(-4.2);
IAI_cap_AP = 10^(-7.2);


%% Channel Parameters
Num_DelayTaps = 4;
Bandwidth = 100*10^6;
Subcarrier_bandwidth = 15 * 10^3;
SymbolDuration = 66.67 * 10^-6;
GuardPeriod = 1 * SymbolDuration;
Carrier_Frequency = 5 * 10^9;
Wavelength = 3*10^8 / Carrier_Frequency;
No_dB = -174+10*log10(Bandwidth); % Noise power in dBm, -174 dBm/Hz
No = 10^(.1*No_dB) * 10^-3;    % Noise power absolute

filename_1 = ['test_',num2str(Num_AP),num2str(Num_MS),'.mat']; %% train or test
load(filename_1)

ii = 0;
while ii<=999
    tic
    ii = ii + 1;
    %for ii = 1:100
    Beta_MS_MS = squeeze(MM(ii,:,:));
    Beta_AP_AP = squeeze(AA(ii,:,:));
    Omega = squeeze(f_zf(ii,:,:,:));
    Upsilon = squeeze(w_zf(ii,:,:,:));
    p_final = zeros(Num_MS,Num_DLsubcarrier+Num_ULsubcarrier,Num_AP);
    for l = 1:Num_AP
        tmp = squeeze(Omega(l,:,:)).';
        P(l,:) = tmp(:).';
    end
    P = (P.^2) ./ (SI_cap_MS*Power_MS_W+No);
    P = waterfill(Power_AP_W, 1./P);
    p_UL_ini = (Power_MS_W / Num_ULsubcarrier) .* ones(Num_MS,Num_ULsubcarrier);
    for l = 1:Num_AP
        tmp = reshape(P(l,:),Num_DLsubcarrier,Num_MS);
        p_final(:,1:Num_DLsubcarrier,l) = tmp.';
    end
    
    P_2 = squeeze(sum(Upsilon,1));
    P_2 = Num_AP^2 ./ ( P_2 *(SI_cap_AP* Power_AP_W + No ));
    P_2 = waterfill(Power_MS_W, 1./P_2);
    p_final(:,Num_DLsubcarrier+1:end,1) = P_2;
    
    [R_AfW,R_AfW_MS] = MDD_SumRateCalculate(p_final,Omega,Upsilon,SI_cap_MS,SI_cap_AP,No,Num_AP,Num_MS,Num_DLsubcarrier,Num_ULsubcarrier,Num_Sumsubcarrier,...
        Beta_AP_AP,Beta_MS_MS,IAI_cap_AP,IMI_cap_MS);
    
    SE(ii) = R_AfW;

end

filename = sprintf('GNN_ave_246.txt');
fid1 = fopen(filename,'at+');
fprintf(fid1,'%6.6f ',SE);
fprintf(fid1,'\n');
fclose(fid1);

