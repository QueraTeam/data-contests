clc
clear
close all
%%
NI=1; % only one image attached for the test. the real training
       % has been done by more images
for i=1:NI
    i
    strin=['dataset2/in/1 (',num2str(i),').jpg'];
    strout=['dataset2/out/Label_',num2str(i),'.png'];
    Iin=imread(strin);
    Iout=imread(strout);
    Iin=imresize(Iin,[300,1000]);
    Iout=imresize(Iout,[300,1000]);
    Iin180=imrotate(Iin,180);
    Iout180=imrotate(Iout,180);
    Iinlr=fliplr(Iin);
    Ioutlr=fliplr(Iout);
    Iinud=flipud(Iin);
    Ioutud=flipud(Iout);
    Iinscale=imresize(imresize(Iin,0.5),[300,1000]);
    Ioutscale=Iout;
    
    IinCONT=histeq(Iin);
    IoutCONT=Iout;
    
    imwrite(Iin,['Data/in/',num2str(i),'.jpg'])
    imwrite(Iout,['Data/out/',num2str(i),'.jpg'])
    imwrite(IinCONT,['Data/in/',num2str(i+1*NI),'.jpg'])
    imwrite(IoutCONT,['Data/out/',num2str(i+1*NI),'.jpg'])
    
    imwrite(Iin180,['Data/in/',num2str(i+2*NI),'.jpg'])
    imwrite(Iout180,['Data/out/',num2str(i+2*NI),'.jpg'])
    imwrite(Iinlr,['Data/in/',num2str(i+3*NI),'.jpg'])
    imwrite(Ioutlr,['Data/out/',num2str(i+3*NI),'.jpg'])
    imwrite(Iinud,['Data/in/',num2str(i+4*NI),'.jpg'])
    imwrite(Ioutud,['Data/out/',num2str(i+4*NI),'.jpg'])
    imwrite(Iinscale,['Data/in/',num2str(i+5*NI),'.jpg'])
    imwrite(Ioutscale,['Data/out/',num2str(i+5*NI),'.jpg'])
end