%% reading image
clear all 
close all 
clc 
global mm nn rr tt 
%[mm nn]=Size Original Image , [rr tt]= size logo 
%rr*tt should be less than mm/3*nn/3 
 
%Reading Original Image 
%[fn fp]=uigetfile('shuiy.jpg.jpg'); 
I=imread('lena.png');
I=rgb2gray(I); 

[mm nn]=size(I); 
figure, subplot(2,2,1),imshow(I) ;
title(' Original Image') ;
 
%Reading LogO 
%[fn fp]=uigetfile('eee.bmp.jpg'); 
logo=imread('logo.jpg');
%logo=rgb2gray(logo); 
logo=imresize(logo,0.1);
[rr tt]=size(logo); 
 
logo=im2bw(logo,.8); 
logo=double(logo); 
subplot(2,2,3),imshow(logo) ;
title(' Original Logo') ;

%% adding water mark
%*****************************embedding***************************** 
 
block=zeros(3,3); 
s_p=zeros(3,3); 
m_p=zeros(3,3); 
blocki=zeros(1,9); 
s_pi=zeros(1,9); 
m_pi=zeros(1,9); 
n=2; 
beta=0.07; 
% Based on paper beta=0.08 
Image=int16(I); 
Image_waterMarked=Image; 
for i=3:3:mm-3 
    for j=3:3:nn-3 
        Image_c=Image(i+1,j+1); 
        f_xor=0; 
        for k=0:2 
            for l=0:2 
                %%%%%%%%%%%%%%%%%%%LBP pattern%%%%%%%%  
                block(k+1,l+1)=Image(i+k,j+l); 
                s_p(k+1,l+1) = Sng(block(k+1,l+1),Image_c); 
                %%%%%%%%%%%%%%%%%%%%%%%% 
                m_p(k+1,l+1)=abs( block(k+1,l+1)-Image_c); 
                %%%%%%%%%%%%%%%%%%%5 
                s_x= s_p(k+1,l+1); 
                f_xor = xor(f_xor,s_x); 
                %%%%%%%%%%%%%%%%%%%%%%%% 
            end 
        end 
        %%%%%%%%%%%%%%%%%%%%%%%%% 
            blocki=block(:); 
            s_pi=s_p(:); 
            m_pi=m_p(:); 
        block_min=min(blocki); 
        m_pi(5)=max(m_pi); 
        m_pmin=min(m_pi); 
        for h=1:9 
            blockit=m_pi(h); 
            if(blockit==m_pmin) 
                x=h; 
            end 
        end 
        index=blocki(x);        
        s_pmin=s_pi(x);       
        %%%%%%%%%%%%%%%%%%%%%%%%embedding%%%%%%%%%%%%%%% 
        if i/3<=rr  
            if j/3<= tt 
        if (f_xor==logo(i/3,j/3)) 
            for m=0:2 
                for n=0:2 
                    Image_waterMarked(i+m,j+n)=block(m+1,n+1); 
                end 
            end 
        elseif s_pmin==1 
            block_min1=(index-m_pmin)*(1-beta); 
            blocki(x)=block_min1; 
            block=reshape(blocki,3,3); 
            for m=0:2 
                for n=0:2 
                    Image_waterMarked(i+m,j+n)=block(m+1,n+1); 
                end 
            end 
             
        elseif(s_pmin==0) 
            block_min2=(index+m_pmin)*(1+beta); 
            blocki(x)=block_min2; 
            block=reshape(blocki,3,3); 
            for m=0:2 
                for n=0:2 
                    Image_waterMarked(i+m,j+n)=block(m+1,n+1); 
                end 
            end 
        end 
            end 
        end 
    end 
end 
 
a=uint8(Image_waterMarked); 
imwrite(a,'Watermarked_Image.jpeg')
subplot(2,2,2), 
imshow(a) 
title(' watermarked image') 
clear i j k l 
%% extract logo
%*************************extraction************************ 
logoextract=extract(a,mm,nn,rr,tt); 
subplot(2,2,4), 
imshow(logoextract) 
title(' extracted logo') 
%% apply attack
%*********************************attack************************ 
global img_crop img_noise img_contrast  
%%%%%%%%%%%%%%%%%% croping %%%%%%%%%%%%%%%%%%%%% 
 
img_crop=a; 
for i=fix(mm/6):fix(mm/2) 
    for   j=fix(nn/6):fix(nn/2) 
        img_crop(i,j)=0; 
    end 
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%niose%%%%%%%%%%%%%% 
 
img_noise=imnoise(a,'salt & pepper',0.07); 
 
%%%%%%%%%%%%%%%%%%%%%contrast%%%%%%%%%%%%%%%%%% 
image=[-100 -70 -50 -30 100 70 50 30];
img_contrast=a+100; 
 
%%%%%%%%%%%%%%%%%%%%%%%%%jpeg%%%%%%%%%%%%%%%%%%%%%5 
 
uu=99; 
imwrite(a,'Compressed.jpg','jpg','quality',uu); 
img_compress=imread('Compressed.jpg'); 
 
%%%%%%%%%%%%%%%%%%%%%%tampered image%%%%%%%%%%%%%%%%% 
 
tamp=imread('T_background.jpg'); 
tamp=rgb2gray(tamp); 
tamp=imresize(tamp,[64,64]);
[zz xx]=size(tamp);
img_tamp=a; 
for i=1:zz 
    for j=1:xx 
        img_tamp(i+128,j+128)=tamp(i,j); 
    end 
end 

 
for i=1:5 
img_input=input(['please enter image attack number:\n1.img_crop\n2.img_noise\n3.img_contrast\n4.img_compress\n5.img_tamp\n']); 
for j=1:5 
    if j==img_input 
if img_input==1 
    img_input=img_crop; 
elseif img_input==2 
    img_input=img_noise; 
elseif img_input==3 
    img_input=img_contrast; 
elseif img_input==4 
    img_input=img_compress; 
elseif img_input==5 
    img_input=img_tamp; 
end 
    subplot(2,2,2), 
imshow(img_input) 
title('Water Marked Image') 
logoextract=extract(img_input,mm,nn,rr,tt); 
subplot(2,2,4), 
imshow(logoextract) 
title('Extracted Logo'),uu=99; 
imwrite(a,'Compressed.jpg','jpg','quality',uu); 
img_compress=imread('Compressed.jpg'); 
figure(10000)
subplot(232),imshow(a),title('watermared image'),subplot(231),imshow(I),title('original image'),subplot(234),imshow(logo),title('original logo'),
subplot(235),imshow(logoextract),title('extracted logo'),subplot(233),imshow(tamp),title('tampering image'),subplot(236),imshow(img_tamp),title('tampered image'),
    else 
        disp ('End'); 
        clc 
         
    end 
end 
end

%% running with multiple noise rate
noise=[0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08];
PSNR=zeros(1,9);
EBR=zeros(1,9);
logos=size(logo,1)*size(logo,2);uu=99; 
imwrite(a,'Compressed.jpg','jpg','quality',uu); 
img_compress=imread('Compressed.jpg'); 

% for i=1:9
%     
% noisyImg = imnoise(a,'salt & pepper',noise(i));  
% imwrite(noisyImg,['noisyImg' num2str(noise(i)) '.jpg']);
% 
% noisyImg=int16(noisyImg);
% logoextract=extract(a,mm,nn,rr,tt); 
% imwrite(a,'Compressed.jpg','jpg','quality',uu); 
% img_compress=imread('Compressed.jpg'); %(noisyImg,mm,nn,rr,tt);
% imwrite(img_compress,['logoExtracted' num2str(noise(i)) '.jpg']);
% EBR(i)=(sum(sum(xor(logo,logoextract))))/logos;
% distortion=noisyImg-Image;
% PSNR(i)=10*log10((mm*nn*255^2)/( sum(sum(distortion.^2))));
% 
% end
% %% drawing figure
% figure, plot(uu,EBR*100);
% title('Relationship EBR and quality level');
% xlabel('Quality')
% ylabel('EBR(%)')
% %% psnr
% figure, plot(uu,PSNR);
% title('Relationship PSNR and quality level');
% xlabel('Quality')
% ylabel('PSNR')
% 
% % image compress
% 
% % running with multiple compression quality
% uu=[100 90 80 70 60 50];
% PSNR=zeros(1,6);
% EBR=zeros(1,6);
% logos=size(logo,1)*size(logo,2);
% 
% for i=1:6
% 
% imwrite(a,['Compress' num2str(uu(i)) '.jpg'],'jpg','quality',uu(i)); 
% img_compress=imread(['Compress' num2str(uu(i)) '.jpg']); 
%    
% logoextract=extract(img_compress,mm,nn,rr,tt); 
% imwrite(logoextract,['logoExtracted Compress' num2str(uu(i)) '.jpg']);
% EBR(i)=(sum(sum(xor(logo,logoextract))))/logos;
% img_compress=int16(img_compress);
% distortion=img_compress-Image;
% PSNR(i)=10*log10((mm*nn*255^2)/( sum(sum(distortion.^2))));
% 
% end
uu=100;
PSNR=zeros(1,9);
c_v=[-100 -70 -50 -30 0 30 50 70 100];
for i=1:9
img_contras=a+c_v(i);
imwrite(img_contras,['contrast' num2str(c_v(i)) '.jpg']);
img_contras=int16(img_contras);
logoextract=extract(a,mm,nn,rr,tt); 
imwrite(a,'Compressed.jpg','jpg','quality',uu); 
img_compress=imread('Compressed.jpg'); %(noisyImg,mm,nn,rr,tt);
imwrite(logoextract,['logoExtracted' num2str(c_v(i)) '.jpg']);
EBR(i)=(sum(sum(xor(logo,logoextract))))/logos;
distortion=img_contras-Image;
PSNR(i)=10*log10((mm*nn*255^2)/( sum(sum(distortion.^2))));
end
%% psnr
figure, 
subplot(211)
p=plot(c_v,PSNR);
p.Marker='o';
title('Relationship PSNR and Contraast level');
xlabel('Contrast')
ylabel('PSNR')
subplot(212)
p1=plot(c_v,EBR*100);
p1.Marker='o';
title('Relationship EBR and Contrast level');
xlabel('Contrast')
ylabel('EBR(%)')