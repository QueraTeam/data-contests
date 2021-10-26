clc
clear
close all
%%
NI=1; % only one image attached for the test. the real training
       % has been done by more images
for i=1:NI
    i
    strempty=['dataset/empty/',num2str(i),'.JPG'];
    strFull=['dataset/full/',num2str(i),'.JPG'];
    Iempty=imread(strempty);
    Ifull=imread(strFull);
    Iempty=imresize(Iempty,[224,224]);
    Ifull=imresize(Ifull,[224,224]);
    Iin180=imrotate(Iempty,180);
    Iout180=imrotate(Ifull,180);
    Iinlr=fliplr(Iempty);
    Ioutlr=fliplr(Ifull);
    Iinud=flipud(Iempty);
    Ioutud=flipud(Ifull);
    Iinscale=imresize(imresize(Iempty,0.5),[224,224]);
    Ioutscale=imresize(imresize(Ifull,0.5),[224,224]);
    
    IinCONT=histeq(Iempty);
    IoutCONT=histeq(Ifull);
    IinCONT2=(Iempty)+5;
    IoutCONT2=(Ifull)+5;
    
    imwrite(Iempty,['Data/empty/',num2str(i),'.jpg'])
    imwrite(Ifull,['Data/full/',num2str(i),'.jpg'])
    imwrite(IinCONT,['Data/empty/',num2str(i+1*NI),'.jpg'])
    imwrite(IoutCONT,['Data/full/',num2str(i+1*NI),'.jpg'])
    
    imwrite(Iin180,['Data/empty/',num2str(i+2*NI),'.jpg'])
    imwrite(Iout180,['Data/full/',num2str(i+2*NI),'.jpg'])
    imwrite(Iinlr,['Data/empty/',num2str(i+3*NI),'.jpg'])
    imwrite(Ioutlr,['Data/full/',num2str(i+3*NI),'.jpg'])
    imwrite(Iinud,['Data/empty/',num2str(i+4*NI),'.jpg'])
    imwrite(Ioutud,['Data/full/',num2str(i+4*NI),'.jpg'])
    imwrite(Iinscale,['Data/empty/',num2str(i+5*NI),'.jpg'])
    imwrite(Ioutscale,['Data/full/',num2str(i+5*NI),'.jpg'])
    imwrite(IinCONT2,['Data/empty/',num2str(i+6*NI),'.jpg'])
    imwrite(IoutCONT2,['Data/full/',num2str(i+6*NI),'.jpg'])

end