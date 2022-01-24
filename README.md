### Blood Donation Prediction with Machine Learning

Analizlerde kullanılmak üzere Tayvan'daki Hsin-Chu şehrindeki Kan Transfüzyon Servis Merkezinin bağışçı veritabanından veri seti temin edilmiştir. 
Kan Transfüzyon Hizmet Merkezi, kan nakli servis otobüsünü yaklaşık üç ayda bir bağışlanan kanı toplamak için Hsin-Chu şehrindeki bir üniversiteye aktarmaktadır.
Kullanılan kan transfüzyon veri setinde, bir model oluşturmak için bağışçı veritabanından rastgele 748 bağışçı seçilmiştir. 
Amaç, kan bağışçısının belirli bir zaman içinde (Mart 2007) kan bağışı yapıp yapmayacağını tahmin etmektir.
## Bu veritabanı RFMTC modelini izlemektedir ve içerdiği nitelikler aşağıda gösterilmiştir.
●	Recency (Yenilik- son bağıştan bu yana geçen ay),
●	Frequency (Sıklık- toplam bağış sayısı),
●	Monetary (Değer- cc olarak bağışlanan toplam kan),
●	Time (İlk bağıştan bu yana geçen süre) ve
●	Class (İkili değişken bağışçının Mart 2007'de kan bağışında bulunup bulunmadığını temsil eder ,1 kan bağışlamak anlamına gelir; 0 bağışçının kan bağışlamadığı anlamına gelir).
Verinin içerisinde boş veya yanlış değerler olup olmadığı kontrol edilir. 
Kullanılan veri setinde, boş veya yanlış veri bulunmadığı için veri ön işleme aşamasında bu işlemler uygulanmamıştır. 
Ayrıca verilerin tamamı nümerik olduğu için tür dönüşümü yapmaya ihtiyaç yoktur.
Kullanılan veri seti, %67 eğitim verisi ve %33 test verisi olarak iki parçaya bölünmüştür. Veri setinin ilk dört kolonu bağımsız değişken olarak belirlenmiştir.
Son kolon olan class kolonu ise, bağımlı değişken olarak belirlenmiştir. 
Bu kolon bağışçının Mart 2007'de kan bağışında bulunup bulunmadığını temsil eder,1 değeri kan bağışlamak anlamına gelirken; 0 değeri kan bağışlamadığı anlamına gelmektedir.
## Modelin Performans Değerlendirmesi
Algoritmaların performansını kıyaslamak için 10-katlı çapraz doğrulama kullanılmıştır. 
Ayrıca karmaşıklık matrisinden elde edilen sonuçlar, ROC eğrisi ve AUC değerleri kullanılarak performans sonuçlarının doğru ve birbiriyle tutarlı sonuçlar verip vermediği incelenmiştir.
# Kullanılan Veri seti:
Açık kaynaklı verilerin bulunduğu [https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center] adresinden alınmıştır.
