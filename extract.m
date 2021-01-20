function logoextract=extract(attack,mm,nn,rr,tt)
Image_attack=int16(attack);
Image_water=Image_attack;
figure;
imshow(Image_water)
watermark=zeros(rr,tt);
block2=zeros(3,3);
for i=3:3:mm-3
    for j=3:3:nn-3
        f_xorwat=0;
        for k=0:2
            for l=0:2
                Image_c1=Image_water(i+1,j+1);
                block2(k+1,l+1)=Image_water(i+k,j+l);
                s_p(k+1,l+1) = Sng(block2(k+1,l+1),Image_c1);
                s_x= s_p(k+1,l+1);
                f_xorwat = xor(f_xorwat,s_x);
            end
        end
        if (f_xorwat==1)
            if i/3<=rr
                if j/3<=tt
            watermark(i/3,j/3)=1;
                end
            end
        else
            if i/3<=rr
                if j/3<=tt
            watermark(i/3,j/3)=0;
                attack(i/3,j/3)=0;
                end
            end
        end
        figure;
        imshow(attack)
        watermark=double(watermark);
    end
end
logoextract=watermark;
