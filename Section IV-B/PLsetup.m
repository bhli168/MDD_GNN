function [Beta_AP_AP,Beta_AP_MS,Beta_MS_MS, AP_AP_PL, MS_MS_PL, AP_MS_PL, AP_location, MS_location] = PLsetup(Num_AP,Num_MS,SquareLength,sha_F)

%% AP MS distribution
while 1
    AP_location = SquareLength * (rand(Num_AP,1) + 1i * rand(Num_AP,1));
    MS_location = SquareLength * (rand(Num_MS,1) + 1i * rand(Num_MS,1));
    
    AP_AP_PL = abs(repmat(AP_location,1,Num_AP) - repmat(AP_location.',Num_AP,1));
    a1 = length(find(AP_AP_PL>0&AP_AP_PL<10));
    MS_MS_PL = abs(repmat(MS_location,1,Num_MS) - repmat(MS_location.',Num_MS,1));
    a2 = length(find(MS_MS_PL>0&MS_MS_PL<10));
    AP_MS_PL = (abs(repmat(AP_location.',Num_MS,1) - repmat(MS_location,1,Num_AP))).';
    a3 = length(find(AP_MS_PL>0&AP_MS_PL<10));
    if a1+a2+a3==0
        break
    end
end
for ll = 1:Num_AP
    for d = 1:Num_MS
        shadow = sha_F * randn();
        Beta_AP_MS(ll,d) = 10 ^ ((-30.5 - 36.7 * log10(AP_MS_PL(ll,d)) + shadow) / 10);
    end
end
for ll = 1:Num_AP
    for lll = 1:Num_AP
        shadow = sha_F * randn();
        if ll == lll
            Beta_AP_AP(ll,lll) = 0;
        else
            Beta_AP_AP(ll,lll) = 10 ^ ((-30.5 - 36.7 * log10(AP_AP_PL(ll,lll)) + shadow) / 10);
        end
        
    end
end
Beta_AP_AP = triu(Beta_AP_AP,1) + triu(Beta_AP_AP,1).';
for dd = 1:Num_MS
    for ddd = 1:Num_MS
        shadow = sha_F * randn();
        if dd == ddd
            Beta_MS_MS(dd,ddd) = 0;
        else
            Beta_MS_MS(dd,ddd) = 10 ^ ((-30.5 - 36.7 * log10(MS_MS_PL(dd,ddd)) + shadow) / 10);
        end
        
    end
end
Beta_MS_MS = triu(Beta_MS_MS,1) + triu(Beta_MS_MS,1).';
end

