function [H_AP_AP_DL,H_AP_MS_DL,H_MS_MS_DL,H_AP_AP_UL,H_AP_MS_UL,H_MS_MS_UL] = MDD_TDCSI_FDCSI(g_AP_AP,g_AP_MS,g_MS_MS,Num_AP,Num_MS,Num_AP_ant,Num_DelayTaps,DLPhi,ULPhi,F,Psih,Num_DLsubcarrier,Num_ULsubcarrier)

H_AP_AP_temp_DL = cell(Num_AP,Num_AP);
H_AP_MS_temp_DL = cell(Num_AP,Num_MS);
H_MS_MS_temp_DL = cell(Num_MS,Num_MS);
H_AP_AP_temp_UL = cell(Num_AP,Num_AP);
H_AP_MS_temp_UL = cell(Num_AP,Num_MS);
H_MS_MS_temp_UL = cell(Num_MS,Num_MS);
H_AP_AP_DL = cell(Num_AP,Num_AP);
H_AP_MS_DL = cell(Num_AP,Num_MS);
H_MS_MS_DL = cell(Num_MS,Num_MS);
H_AP_AP_UL = cell(Num_AP,Num_AP);
H_AP_MS_UL = cell(Num_AP,Num_MS);
H_MS_MS_UL = cell(Num_MS,Num_MS);
MDD_DLsubSet = DLPhi * F * Psih;
MDD_ULsubSet = ULPhi * F * Psih;
for m = 1:Num_AP
    for n = 1:Num_AP
        Temp = reshape(g_AP_AP{m,n},Num_DelayTaps,[]);
        H_AP_AP_temp_DL{m,n} = reshape(MDD_DLsubSet * Temp,[],Num_AP_ant,Num_AP_ant);
        H_AP_AP_temp_UL{m,n} = reshape(MDD_ULsubSet * Temp,[],Num_AP_ant,Num_AP_ant);
        for ds = 1:Num_DLsubcarrier
            H_AP_AP_DL{m,n}(:,:,ds) = reshape(H_AP_AP_temp_DL{m,n}(ds,:,:),Num_AP_ant,Num_AP_ant);
        end
        for us = 1:Num_ULsubcarrier
            H_AP_AP_UL{m,n}(:,:,us) = reshape(H_AP_AP_temp_UL{m,n}(us,:,:),Num_AP_ant,Num_AP_ant);
        end
    end
end

for m = 1:Num_AP
    for n = 1:Num_MS
        H_AP_MS_temp_DL{m,n} = MDD_DLsubSet * g_AP_MS{m,n};
        H_AP_MS_temp_UL{m,n} = MDD_ULsubSet * g_AP_MS{m,n};
        for ds = 1:Num_DLsubcarrier
            H_AP_MS_DL{m,n}(:,:,ds) = H_AP_MS_temp_DL{m,n}(ds,:);
        end
        for us = 1:Num_ULsubcarrier
            H_AP_MS_UL{m,n}(:,:,us) = H_AP_MS_temp_UL{m,n}(us,:);
        end
    end
end

for m = 1:Num_MS
    for n = 1:Num_MS
        H_MS_MS_temp_DL{m,n} = MDD_DLsubSet * g_MS_MS{m,n};
        H_MS_MS_temp_UL{m,n} = MDD_ULsubSet * g_MS_MS{m,n};
        for ds = 1:Num_DLsubcarrier
            H_MS_MS_DL{m,n}(:,:,ds) = H_MS_MS_temp_DL{m,n}(ds,:,:);
        end
        for us = 1:Num_ULsubcarrier
            H_MS_MS_UL{m,n}(:,:,us) = H_MS_MS_temp_UL{m,n}(us,:,:);
        end
    end
end






end