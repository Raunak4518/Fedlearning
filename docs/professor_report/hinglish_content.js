// Content of the personal Hinglish notes (assembled by build_hinglish.py)
const card = (a, b, rows) => table([a, b], rows, [1.7, 7.9], { left: true });
const yaad = (lines) => C.push(...callout("Yaad rakhne wali baatein", lines, "FFF8E6"));
const prof = (qa) => { C.push(H3("Professor pooche to")); qa.forEach(([q, a]) => { C.push(P("**Q: " + q + "**")); C.push(P("A: " + a)); }); };

// ---------------- title
C.push(new Paragraph({ spacing: { before: 2000, after: 200 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Ours-A: har component, aasaan bhasha mein", bold: true, size: 44, color: BLUE })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [new TextRun({ text: "Sirf mere liye notes · kya karta hai, kaise karta hai, kahan lagta hai, kyun kaam karta hai", italics: true, size: 22, color: GREY })] }));
C.push(...table(["#", "Component", "Ek line mein", "Page"], [
  ["0", "Overview", "GeFL-F mein kya badla, ek nazar mein", "2"],
  ["1", "Class statistics (exact sums)", "har client ek baar apne features ka total bhejta hai", "3"],
  ["2", "PC-VAE generator", "class ko learned row se nahi, exact class mean se pehchano", "5"],
  ["3", "MC: moment correction", "fake samples ka mean aur spread exact real ke barabar", "7"],
  ["4", "KH: kernel herding", "4n mein se wo n chuno jinka fingerprint real jaisa ho", "9"],
  ["5", "CSL: consensus soft labels", "label = aadha asli class + aadha sab heads ki raay", "11"],
  ["6", "LA: logit adjustment", "client apni data ki skew na seekhe", "13"],
  ["7", "T_s = 10", "achha generator hai to fake data pe zyada training", "15"],
  ["8", "Privacy (DP) option", "sums mein noise daal ke formal guarantee", "17"],
], [0.4, 2.6, 5.4, 0.8]));

// ---------------- 0 overview
C.push(H1("0. Overview: GeFL-F mein humne kya badla"));
C.push(P("**GeFL-F ek line mein:** sab clients ek chhota feature extractor (FE) saath mein train karte hain (Stage I), phir ek generator saath mein train karte hain jo kisi bhi class ke fake features bana sake (Stage II), phir har client apna private head train karta hai, pehle fake features pe, phir apne real features pe (Stage III)."));
C.push(P("**Problem:** jab clients ka data long-tailed hota hai (kuch classes bahut kam), to GeFL-F 21–25 points gir jaata hai. Rare classes almost predict hi nahi hoti."));
C.push(P("**Asli wajah:** generator ke andar class ki pehchaan ek learned row w_y mein hoti hai. FedAvg us row ko average karta hai, aur jo clients ke paas wo class nahi hai, wo row ko zero ki taraf kheenchte hain. Rare class = zyada clients ke paas nahi = row shrink = fake features bekaar."));
C.push(P("**Humara main idea:** class ki jaankari FedAvg se *estimate* mat karo. Secure aggregation se clients ke sums ka *exact* total mil sakta hai. Class information wahin se lo."));
C.push(...img("f07_esa.png", 600, "Ours-A inside GeFL-F. Blue boxes = humare changes."));
C.push(...table(["Step", "MNIST", "FMNIST", "Kya fix hua"], [
  ["GeFL-F", "75.25", "58.47", "—"],
  ["+ LA", "78.92", "64.78", "client ki apni skew"],
  ["+ HWA (beech ka step, baad mein hata diya)", "87.89", "76.02", "rare rows shrink nahi hoti"],
  ["+ CSL", "90.40", "76.60", "fake features ke labels behtar"],
  ["PC-VAE (HWA ki jagah)", "90.75", "75.96", "class rows hi nahi"],
  ["+ MC", "92.31", "77.56", "exact mean + spread"],
  ["+ KH", "93.11", "78.22", "fake ≈ real"],
  ["+ T_s = 10 = **Ours-A**", "**94.82**", "**79.67**", "achhe generator ka zyada use"],
], [3.8, 1.1, 1.1, 3.6], { highlight: [7], cap: "Long tail, balanced accuracy %, 3 seeds, same splits." }));

// ---------------- 1 statistics
C.push(H1("1. Class statistics: exact sums (secure aggregation)"));
C.push(...callout("Ek line mein", ["Stage I ke baad har client **ek baar** apni har class ke features ka **total (sum)** bhejta hai. Server ko sirf **sab clients ka combined total** milta hai, kisi ek client ka nahi."]));
C.push(H2("Problem kya tha"));
C.push(P("GeFL-F mein class ke baare mein jo bhi pata hai, wo generator ke weights mein chhupa hai, aur weights FedAvg se average hote hain. Rare class ke liye ye average galat ho jaata hai (Section 2 dekho). Humein ek aisi cheez chahiye thi jo **exact** ho, estimate nahi, aur jisse privacy bhi na toote."));
C.push(H2("Simple example"));
C.push(P("Maan lo 10 schools hain aur humein class 9 ke bachchon ki **average height** chahiye. Har school bata de: \"humare paas class 9 ke kitne bachche hain\" aur \"unki height ka total kya hai\". Total ÷ count = exact average. Kisi ek bachche ki height kisi ko pata nahi chali. Bilkul yahi hum features ke saath karte hain."));
C.push(H2("Kya bhejte hain"));
C.push(...table(["Har client k, har class c ke liye", "Size", "Server kya banata hai", "Kis kaam aata hai"], [
  ["n_kc = kitne samples", "1", "n_c = sab ka total", "class size"],
  ["S_kc = Σ h (features ka sum)", "768", "m_c = S_c / n_c  (class mean)", "PC-VAE (Stage II) aur MC"],
  ["Q_kc = Σ ‖h‖²", "1", "V_c = Q_c / n_c − ‖m_c‖²  (spread)", "MC"],
  ["Φ_kc = Σ φ(h)  (random Fourier features)", "6,144", "μ_c = Φ_c / n_c  (fingerprint)", "KH"],
], [3.0, 0.8, 3.0, 2.8]));
C.push(eq("eq_stats"));
C.push(H2("Kaise kaam karta hai (step by step)"));
C.push(B("**Step 1:** Stage I khatam, FE frozen. Ab har image ka feature h (3×16×16 = 768 numbers) kabhi nahi badlega, isliye ek hi upload kaafi hai."));
C.push(B("**Step 2:** client apne har sample ka h nikalta hai aur class-wise jodta hai: count, sum of h, sum of ‖h‖², sum of φ(h)."));
C.push(B("**Step 3: secure aggregation.** Har client apne numbers mein ek random mask jodta hai. Masks aise banaye jaate hain ki sab clients ke masks ka total zero ho. Server sab add karta hai: masks cancel, sirf asli total bachta hai (Bonawitz et al., CCS 2017)."));
C.push(B("**Step 4:** server divide karke m_c, V_c, μ_c banata hai."));
C.push(H2("Fingerprint φ(h) kya hai"));
C.push(P("φ(h) = √(2/D) · cos(W h + b), D = 6,144 random directions (3 alag widths ke kernels). Iska average μ_c poori class ki **shape** ko describe karta hai, sirf centre ko nahi. Do classes ka fingerprint same hai to unka distribution bhi same hai (characteristic kernel). Bonus: ‖φ‖ ≤ √2, to ek sample ka effect limited hai, DP ke liye perfect."));
C.push(H2("Architecture mein kahan"));
C.push(...card("Detail", "Value", [
  ["**Kab**", "Ek baar, Stage I ke turant baad"],
  ["**Kahan**", "Client calculate karta hai, server sirf total dekhta hai"],
  ["**Size**", "10 × (1 + 768 + 1 + 6,144) ≈ 69,000 numbers ≈ **0.3 MB per client, sirf ek baar** (generator to har round 22 MB bhejta hai)"],
  ["**Kaun use karta hai**", "m_c → PC-VAE aur MC · V_c → MC · μ_c → KH"],
  ["**Code**", "anchor_stats (experiments/core.py:1132), federated_kme (core.py:1459), RFFMap (core.py:1181)"],
]));
C.push(H2("Kyun kaam karta hai"));
C.push(P("Kyunki ye **exact** hai. Kuch learn nahi ho raha, kuch average nahi ho raha, isliye FedAvg isse shrink nahi kar sakta. Rare class ke sirf 24 samples hon tab bhi uska mean bilkul sahi aata hai."));
prof([
  ["Ye privacy leak nahi hai?", "Server ko sirf sab clients ka total milta hai (secure aggregation), FedProto bhi class prototypes share karta hai. Formal guarantee chahiye to noise daal sakte hain: ε = 8 pe MNIST 90.36 (GeFL-F 75.25 se ab bhi +15)."],
  ["Ek hi baar kyun?", "FE Stage I ke baad frozen hai, features badalte hi nahi."],
]);
yaad(["1) Sirf sums, sirf ek baar, 0.3 MB.", "2) Teen cheezein milti hain: mean m_c, spread V_c, fingerprint μ_c.", "3) Exact hai, isliye FedAvg ki shrink problem isme nahi hoti."]);

// ---------------- 2 PC-VAE
C.push(H1("2. PC-VAE: class ko exact mean se pehchano"));
C.push(...callout("Ek line mein", ["Naya generator jo class ko ek **learned row** se nahi, balki **exact class mean m_y** se pehchanta hai. Isme koi class-specific weight hai hi nahi, to FedAvg ke paas shrink karne ko kuch nahi."]));
C.push(H2("Problem kya tha (GeFL-F ka CVAE-F)"));
C.push(P("CVAE-F ke decoder ka input = Linear(z) + **w_y**, jahan w_y ek 10 × 512 table ki row hai. Ab socho class c jiske paas nahi hai, us client pe kya hota hai:"));
C.push(B("Uske data mein class c hai hi nahi, to w_c pe koi data gradient nahi aata. Sirf weight decay ka gradient aata hai, jo row ko zero ki taraf le jaata hai."));
C.push(B("Adam har coordinate ka step normalise karta hai, to ye chhota sa decay bhi **poora step** ban jaata hai (lagbhag η · sign(w_c))."));
C.push(B("Holders row ko upar kheenchte hain, non-holders neeche. FedAvg average karta hai. Agar class c ke holders aadhe se kam (m_c/K < ½), to row **har round** chhoti hoti hai (Proposition 1)."));
C.push(eq("eq_collapse"));
C.push(P("**Proof jo humne naapa:** rare/common row size ratio MNIST pe 0.33, FMNIST pe 0.22. Real-data referee GeFL-F ke fake rare-class samples mein se sirf **11 %** ko sahi class bolta hai (MNIST)."));
C.push(...img("f05_problem.png", 560, "GeFL-F ki problem: accuracy, rows ka shrink, aur Proposition 1 ka bound."));
C.push(H2("Pehli koshish: HWA (aur kyun hataya)"));
C.push(P("HWA mein row c ko sirf un clients pe average karte the jinke paas class c hai. Row ratio 0.80–1.01 pe wapas aa gaya, MNIST +9.0. Lekin row abhi bhi learned thi, counts chahiye the, aur 100 clients pe har client ke paas rare class ke bahut kam samples the, to accuracy 4.8 gir gayi. Isliye humne rows ko hi hata diya."));
C.push(H2("PC-VAE kaise kaam karta hai"));
C.push(...img("f13_pcvae.png", 600, "PC-VAE ke saare shapes. Blue boxes = jahan exact class mean aata hai."));
C.push(B("**Encoder input:** [h ; m_y] = 3 + 3 channels = 6×16×16. Encoder: 4 × (Conv 4×4 stride 2 → BN → ReLU): 64×8×8 → 128×4×4 → 256×2×2 → 512×1×1 → μ, log σ² (16 each)."));
C.push(B("**Decoder input:** Linear(16→512)(z) **+ A · m_y**, jahan A = Linear(768→512) **sab classes ke liye ek hi** hai. Phir ReLU → 4 transposed conv → 3×16×16."));
C.push(B("**Output:** h̃ = ReLU( m_y + decoder output ). Matlab generator sirf ye seekhta hai ki ek sample apni class ke mean se **kitna alag** hai."));
C.push(eq("eq_pc"));
C.push(H2("Simple example"));
C.push(P("CVAE-F: har artist ko har class ka chehra yaad rakhna padta hai (row). Jo class kam dikhi, uska chehra bhool jaata hai. PC-VAE: artist ko class ki **average photo** de do, wo bas ye seekhe ki \"ek insaan average se kaise alag hota hai\" (smile, angle, lighting). Ye skill sab classes mein same hai, to rare class bhi common classes se seekh leti hai."));
C.push(H2("Architecture mein kahan"));
C.push(...card("Detail", "Value", [
  ["**Stage**", "Stage II (CVAE-F ki jagah). Training same: loss ‖h̃ − h‖² + KL, Adam lr 0.001, wd 0.001, batch 64, 5 local epochs, 100 rounds"],
  ["**Aggregation**", "Saare weights normal FedAvg. HWA ya counts ki zaroorat nahi"],
  ["**Generation**", "Class y aur z ~ N(0, I) do, sirf decoder chalao. m_y server ke paas hai"],
  ["**Size**", "5.94 M parameters (CVAE-F 5.55 M)"],
  ["**Code**", "ProtoCVAE (experiments/core.py:353)"],
]));
C.push(H2("Result"));
C.push(B("Rare-class fidelity (referee kitne fake samples ko sahi class bolta hai): MNIST 0.11 → **0.91**, FMNIST 0.33 → **0.87**."));
C.push(B("Akele accuracy mein HWA ke barabar (MNIST 90.40 vs 90.75; FMNIST 76.60 vs 75.96). Asli fayda: MC aur KH sirf tab kaam karte hain jab fake classes sahi hon, aur 100 clients pe Ours-A sirf 0.5 girta hai (HWA 4.8)."));
C.push(B("Kami: samples mean ke bahut paas aate hain (spread kam). Isi liye agla step MC hai."));
prof([
  ["Idea kahan se aaya?", "FedProto (Tan et al., AAAI 2022) class prototypes share karta hai. Hum prototype ko generator ki conditioning bana rahe hain aur residual seekh rahe hain."],
  ["SVHN pe kyun kaam nahi karta?", "SVHN ke class means almost same hain (nearest-mean accuracy 12 %, chance 10 %). Mean se class pehchani hi nahi jaati. Ye humari limitation hai, future scope mein fix hai."],
]);
yaad(["1) Problem: learned row w_y + FedAvg + weight decay = rare row shrink.", "2) Fix: row hatao, exact mean m_y se condition karo, output = m_y + residual.", "3) Rare-class fidelity 11 % → 91 % (MNIST)."]);

// ---------------- 3 MC
C.push(H1("3. MC: moment correction"));
C.push(...callout("Ek line mein", ["Har class ke fake samples ko **shift aur stretch** karo taaki unka **mean** aur **spread** bilkul real class ke barabar ho jaaye."]));
C.push(H2("Problem kya tha"));
C.push(P("VAE ke samples \"too clean\" hote hain: sab mean ke paas, variety kam (under-dispersed). Aur rare class ka mean thoda idhar-udhar ho sakta hai. Head agar too-clean fake data pe train hua, to real data ki variety pe confuse hota hai."));
C.push(H2("Kaise kaam karta hai"));
C.push(B("**Step 1:** class c ke 500 fake samples banao. Unka mean m̃_c aur spread Ṽ_c nikalo."));
C.push(B("**Step 2:** scale s_c = √(V_c / Ṽ_c) (0.1 se 10 ke beech clamp). V_c exact real spread hai (Section 1)."));
C.push(B("**Step 3:** ab har naya fake sample x:  x′ = ReLU( m_c + s_c · (x − m̃_c) )."));
C.push(eq("eq_mc"));
C.push(H2("Simple example"));
C.push(P("Marks normalisation jaisa: ek section ke marks ka average 60 aur spread kam hai, board ka average 70 aur spread zyada. Har student ke marks se section ka average ghatao, stretch karo, board ka average jodo. Ranking (shape) same rehti hai, bas centre aur width sahi ho jaate hain."));
C.push(...img("f15_mc_kh.png", 600, "2-D illustration (toy data): (a) generator, (b) MC ke baad sahi centre aur width, (c) KH ke baad sahi shape. Real data waala plot D01 experiment se aayega."));
C.push(H2("Kyun ye hi formula (Proposition 3)"));
C.push(P("Aise bahut saare maps hain jo exact mean aur spread de dein. Humara map unme se wo hai jo samples ko **sabse kam hilata hai** (E‖T(X) − X‖² minimum). Proof Cauchy–Schwarz se: cross term maximum tab hota hai jab naya sample purane ka scaled version ho. Matlab correction ho jaata hai, aur generator ne jo shape seekhi thi wo kharab nahi hoti."));
C.push(H2("Architecture mein kahan"));
C.push(...card("Detail", "Value", [
  ["**Stage**", "Stage II ke baad, Stage III se pehle; sampler ke andar (generator ke upar ek wrapper)"],
  ["**Kahan**", "Server (jiske paas generator aur statistics hain)"],
  ["**Training**", "Koi nahi. Har class ke liye sirf 2 cheezein: shift (m̃_c → m_c) aur scale s_c"],
  ["**Input → output**", "fake h̃ (3×16×16) → corrected h̃ (3×16×16)"],
  ["**Code**", "MomentCalibratedGen (experiments/core.py:1516)"],
]));
C.push(H2("Result"));
C.push(B("PC-VAE ke upar: MNIST 90.75 → **92.31 (+1.56)**, FMNIST 75.96 → **77.56 (+1.60)**."));
C.push(B("Optimal transport ka scalar case hai (Olkin and Pukelsheim, 1982)."));
C.push(B("Limit: sirf total spread (ek number) match hota hai, poori covariance nahi. Covariance version (MC-S / MC-B) future scope hai."));
prof([
  ["ReLU kyun lagaya?", "Real features ReLU ke baad aate hain, to negative nahi ho sakte. Fake features bhi wahi range mein rehne chahiye."],
  ["Spread ko scalar kyun rakha?", "Rare class ke sirf 24 samples hain; 768×768 covariance ka estimate bahut noisy hota. Scalar spread 24 samples se bhi stable hai."],
]);
yaad(["1) VAE samples too clean → MC unhe exact mean + spread pe le aata hai.", "2) Formula: x′ = m_c + √(V_c/Ṽ_c)(x − m̃_c), sabse kam movement.", "3) +1.6 points, koi training nahi."]);

// ---------------- 4 KH
C.push(H1("4. KH: kernel herding (fingerprint se samples chuno)"));
C.push(...callout("Ek line mein", ["Har class ke **4n** fake samples banao, aur unme se wo **n** chuno jinka combined **fingerprint** real class ke fingerprint μ_c se sabse zyada milta ho."]));
C.push(H2("Problem kya tha"));
C.push(P("MC ke baad mean aur spread sahi hain, lekin **shape** abhi bhi galat ho sakti hai: jaise fake data mein ek extra chhota cluster ho jo real mein nahi hai, ya real ka koi hissa fake mein missing ho. Mean aur spread ye nahi pakad sakte."));
C.push(H2("Fingerprint aur MMD"));
C.push(P("Section 1 ka μ_c = class ke saare φ(h) ka average. Ye poore distribution ka fingerprint hai (kernel mean embedding). Do distributions ke fingerprints ki doori ko **MMD** kehte hain. MMD = 0 matlab dono distributions same."));
C.push(H2("Algorithm (greedy)"));
C.push(B("**Step 1:** 4n candidates ka φ(x_i) nikalo (har ek 6,144 numbers). Shuru mein w = μ_c."));
C.push(B("**Step 2:** jo candidate abhi tak nahi liya, unme se wo chuno jiska ⟨φ(x_i), w⟩ sabse bada ho."));
C.push(B("**Step 3:** w ← w + μ_c − φ(x_chosen). Ab w batata hai \"abhi kya missing hai\". Step 2 pe wapas jao, jab tak n samples na ho jaayein."));
C.push(eq("eq_kh"));
C.push(H2("Simple example"));
C.push(P("State ki cricket team chunni hai jo poore state ke players ko represent kare. Pehle sabse typical player lo. Phir dekho kya kami hai (abhi tak sab batsman hain) aur agla player wo lo jo wo kami bhare (bowler). Har pick \"jo missing hai\" ko bharta hai. Random picking mein ye guarantee nahi hoti."));
C.push(H2("Kyun kaam karta hai (Lemma 1)"));
C.push(P("Head ka real data pe error aur fake data pe error ka farak ≤ B · MMD(fake, real). Head fake data pe train hota hai aur real pe test, to MMD kam karna = real accuracy ki guarantee behtar karna. Herding MMD ko random sampling se tez ghataata hai (lagbhag 1/n vs 1/√n)."));
C.push(H2("Architecture mein kahan"));
C.push(...card("Detail", "Value", [
  ["**Stage**", "Stage III se pehle, sampler mein, MC ke baad (generator → MC → KH)"],
  ["**Kahan**", "Server"],
  ["**Uses**", "μ_c (Section 1 ka fingerprint) aur wahi φ map"],
  ["**Training**", "Koi nahi. 4× sampling + har pick pe ek matrix product"],
  ["**Code**", "HerdedGen (experiments/core.py:1470), federated_kme (core.py:1459)"],
]));
C.push(H2("Result"));
C.push(B("MC ke upar: MNIST 92.31 → **93.11 (+0.80)**, FMNIST 77.56 → **78.22 (+0.66)**."));
C.push(B("**9 out of 9** paired tests mein help kiya (p = 0.004)."));
C.push(B("Interesting: referee fidelity thodi giri (0.94 → 0.88). Ye achha hai: fake features ab real jitne \"hard\" hain, artificially easy nahi."));
C.push(B("Source: Chen, Welling and Smola, Kernel Herding (UAI 2010); MMD: Gretton et al. (JMLR 2012)."));
prof([
  ["Sirf zyada samples kyun nahi bana lete?", "Zyada random samples se bhi wahi galat shape aati hai. Herding galat shape waale samples ko chhod deta hai."],
  ["Fingerprint ko train karna padta hai?", "Nahi. W aur b random fixed hain (seed), sab clients same map use karte hain."],
]);
yaad(["1) Mean + spread sahi, shape galat ho sakti hai → KH.", "2) 4n banao, n chuno, har pick 'missing' ko bharta hai.", "3) MMD kam = real accuracy ki guarantee (Lemma 1); 9/9 tests mein fayda."]);

// ---------------- 5 CSL
C.push(H1("5. CSL: consensus soft labels"));
C.push(...callout("Ek line mein", ["Fake sample ka training label sirf \"ye class y hai\" nahi, balki **aadha y + aadha sab 10 architectures ke heads ki average raay**."]));
C.push(H2("Problem kya tha"));
C.push(P("Generator ne bola \"ye 4 hai\", lekin fake sample thoda 9 jaisa bhi lag sakta hai. One-hot label head ko 100 % confident hone pe majboor karta hai, chahe sample ambiguous ho. Aur 10 alag architectures ke heads ek doosre se kuch nahi seekhte."));
C.push(H2("Kaise kaam karta hai"));
C.push(B("**Step 1:** har round server ek shared fake pool banata hai (PC-VAE → MC → KH)."));
C.push(B("**Step 2:** har client ka head pool pe softmax predict karta hai. Clients sirf **prediction ka sum** bhejte hain (≈ 24 KB per fake epoch)."));
C.push(B("**Step 3:** consensus = 10 architectures ka average softmax."));
C.push(B("**Step 4:** target t = 0.5 · one-hot(y) + 0.5 · consensus. Loss = soft cross-entropy: −Σ t · log p."));
C.push(eq("eq_csl"));
C.push(...img("f14_round3.png", 600, "Ek Stage III round: pool → consensus → 10 fake epochs → 5 real epochs → upload."));
C.push(H2("Simple example"));
C.push(P("Exam ki answer key (generator ka label) kabhi kabhi galat bhi hoti hai. Agar 10 alag tarah se padhne waale achhe students ka average answer bhi dekho, to galti cancel ho jaati hai. CSL = aadha answer key, aadha in 10 students ki raay."));
C.push(H2("Kyun kaam karta hai"));
C.push(B("**Ensemble:** alag architectures alag galtiyan karte hain, average mein cancel hoti hain (knowledge distillation, FedDF)."));
C.push(B("**Soft label:** \"70 % 4, 20 % 9\" jaisi info class similarity sikhati hai (dark knowledge), aur label smoothing jaisa regularise karti hai."));
C.push(B("**β = 0.5:** sirf consensus (β = 1) pe heads khud ki galti copy karte hain; aadha label rakhne se bias aur variance balance hote hain. Test mein β = 0.5 > β = 1."));
C.push(H2("Architecture mein kahan"));
C.push(...card("Detail", "Value", [
  ["**Stage**", "Stage III, synthetic phase, har round"],
  ["**Kahan**", "Clients predict karte hain, server sums jodta hai (secure aggregation)"],
  ["**Communication**", "≈ 24 KB per fake epoch; real data pe koi output share nahi hota"],
  ["**Code**", "stage_iii ka CSL block (experiments/core.py:1774)"],
]));
C.push(H2("Result"));
C.push(B("MNIST long tail +2.5 (87.89 → 90.40), FMNIST +0.6. IID mein bhi +0.5 to +0.8 (MNIST 96.82 vs 96.01; CIFAR-10 59.85 vs 59.10; SVHN 76.30 vs 75.82)."));
C.push(B("Bayesian CSL try kiya, fail hua (heads generator se hi seekhte hain to wo hamesha agree karte hain, ρ̂ → 0.999)."));
C.push(B("Source: FedMD (Li and Wang, 2019); FedDF (Lin et al., NeurIPS 2020)."));
prof([
  ["Privacy?", "Sirf fake features pe predictions ka sum jaata hai. Real data ka kuch bhi nahi."],
  ["Consensus kis round ke heads se?", "Pichhle round ke averaged heads se, round ke shuru mein."],
]);
yaad(["1) Target = ½ one-hot + ½ sab architectures ka average.", "2) Galtiyan cancel + soft info + regularisation.", "3) +2.5 MNIST LT, IID mein bhi fayda."]);

// ---------------- 6 LA
C.push(H1("6. LA: logit adjustment (client ki skew hatao)"));
C.push(...callout("Ek line mein", ["Real data pe training ke waqt logits mein **client ka apna log-prior** jod do. Isse head apne data ki skew (\"jab doubt ho, common class bolo\") nahi seekhta."]));
C.push(H2("Problem kya tha"));
C.push(P("Ek client ke paas 90 % class 0 hai. Uska head seekh leta hai \"doubt ho to 0 bolo\". Test set balanced hai, to rare classes ki recall gir jaati hai. GeFL-F mein fake epochs balanced hote hain, lekin uske baad ke **5 real epochs** head ko wapas client ki skew ki taraf kheench lete hain."));
C.push(H2("Kaise kaam karta hai"));
C.push(B("**Step 1:** client apna prior nikalta hai: π_k(c) = (n_kc + 1) / Σ (n_kc + 1). +1 isliye ki zero class ka log −∞ na ho."));
C.push(B("**Step 2:** training loss = CE( f(h) + τ · log π_k , y ), τ = 1."));
C.push(B("**Step 3:** test pe sirf f(h) use karo (offset hata do)."));
C.push(eq("eq_la"));
C.push(H2("Kyun kaam karta hai"));
C.push(P("Bayes rule: p_train(y | x) ∝ p(x | y) · π(y). Training mein hum f + log π ko p_train(y|x) jaisa banate hain, to **f khud** sirf p(x | y) waala hissa seekhta hai, prior waala hissa log π sambhal leta hai. Test pe log π hatane ka matlab hai uniform prior pe predict karna, jo balanced test ke liye sahi hai."));
C.push(H2("Simple example"));
C.push(P("Ek umpire jaanta hai ki ek team har ball pe appeal karti hai. Wo unki appeal ko thoda discount karta hai. LA head ko batata hai: \"tumhare data mein class 0 bahut hai, isliye uska advantage training mein pehle se de diya; ab tum sirf feature dekh ke faisla seekho\"."));
C.push(H2("Architecture mein kahan"));
C.push(...card("Detail", "Value", [
  ["**Stage**", "Stage III, real-data phase (5 epochs), har round, har client"],
  ["**Communication**", "Kuch nahi. π_k client se bahar nahi jaata"],
  ["**Fake phase pe kyun nahi**", "Fake pool ke labels uniform hain, wahan prior pehle se balanced hai"],
  ["**Code**", "ce_or_la (experiments/core.py:692), local_log_prior (core.py:698)"],
]));
C.push(H2("Result"));
C.push(B("GeFL-F ke upar: MNIST 75.25 → **78.92 (+3.67)**, FMNIST 58.47 → **64.78 (+6.31)**; chaaron datasets pe +3.6 se +6.3."));
C.push(B("IID mein koi effect nahi (π_k uniform hai, log π constant), to balanced setting kharab nahi hoti."));
C.push(B("Source: Menon et al., Logit Adjustment (ICLR 2021); FL mein FedLC (Zhang et al., ICML 2022)."));
prof([
  ["Ye to existing technique hai, novelty kya?", "Haan, LA established hai, hum claim nahi karte. Hum dikhate hain ki GeFL-F mein ye zaroori kyun hai (real phase skew wapas lata hai), aur ye baaki components ke saath add hota hai."],
  ["τ = 1 kyun?", "Theory mein τ = 1 Bayes-optimal hai; humne wahi rakha."],
]);
yaad(["1) Loss mein + log π_k, test pe hata do.", "2) Head prior nahi, feature seekhta hai.", "3) +3.6 to +6.3 sab datasets pe, zero communication."]);

// ---------------- 7 Ts
C.push(H1("7. T_s = 10: achhe generator ka zyada use"));
C.push(...callout("Ek line mein", ["Har Stage III round mein fake data pe **1 ki jagah 10 epochs**, kyunki ab fake features itne achhe hain ki zyada training fayda deti hai."]));
C.push(H2("Problem kya tha"));
C.push(P("GeFL-F T_s = 1 use karta hai. Wajah: uske fake features (khaas kar rare classes ke) bekaar hain. Bekaar data pe zyada training = galat cheez zyada seekhna. Isliye original generator ke saath T_s badhane se kuch nahi milta."));
C.push(H2("Theory (Proposition 5)"));
C.push(P("Ek simple model: head ka error = variance (zyada data se ghatta hai) + bias² (generator ke labels ki galti b). Fake data ko weight w do to error = σ²(1 − w)/n_r + w² b². Iska minimum w* = σ²/(2 n_r b²). Matlab: **generator jitna sahi (b chhota), utna zyada fake data use karna chahiye.**"));
C.push(eq("eq_budget"));
C.push(H2("Kaise lagaya"));
C.push(...card("Detail", "Value", [
  ["**Stage**", "Stage III, har round, har client"],
  ["**Ek fake epoch**", "Client ke real data jitna size, labels uniform (10 classes barabar)"],
  ["**Order**", "10 fake epochs (CSL) → 5 real epochs (LA)"],
  ["**Cost**", "Stage III compute ≈ 5–6× (MNIST 109 s → 680 s); communication sirf CSL sums × 10"],
]));
C.push(H2("Result (humne theory ko test kiya)"));
C.push(...table(["Dataset (long tail)", "Generator", "T_s = 1", "T_s = 10", "Farak"], [
  ["MNIST", "CVAE-F + HWA", "90.20", "90.15", "−0.05"],
  ["MNIST", "**Ours-A (PC + MC + KH)**", "93.11", "**94.82**", "**+1.71**"],
  ["MNIST, 100 clients", "Ours-A", "", "", "+3.9"],
  ["FashionMNIST", "CVAE-F + HWA", "76.94", "79.23", "+2.29"],
  ["FashionMNIST", "Ours-A", "78.22", "79.67", "+1.45"],
  ["CIFAR-10", "Ours-A", "46.06", "44.38", "−1.68 (peak +1.3)"],
  ["SVHN", "Ours-A", "55.89", "54.41", "−1.48 (peak +0.7)"],
], [2.2, 2.6, 1.3, 1.3, 2.2], { highlight: [1] }));
C.push(P("MNIST pe theory exactly dikhti hai: kharab generator ke saath 0, achhe ke saath +1.7. FMNIST pe dono ko fayda hai. CIFAR-10 aur SVHN pe fake features kam faithful hain, to peak accuracy upar jaati hai lekin final round ki accuracy neeche. **Honest baat:** T_s = 10 MNIST/FMNIST pe best hai, CIFAR/SVHN pe nahi."));
prof([
  ["CIFAR pe kyun gira?", "Wahan generator fidelity kam hai (tail fidelity ≈ 40 %), to b bada hai; Proposition 5 yahi kehta hai."],
  ["Kya T_s automatic choose ho sakta hai?", "Haan, future scope: generator ki fidelity se T_s choose karna."],
]);
yaad(["1) Kharab generator → kam fake training; achha → zyada.", "2) MNIST: HWA gen −0.05, Ours-A gen +1.71; K = 100 pe +3.9.", "3) CIFAR/SVHN pe final accuracy giri: honest limitation."]);

// ---------------- 8 DP
C.push(H1("8. Privacy option: differential privacy (DP)"));
C.push(...callout("Ek line mein", ["Agar formal guarantee chahiye, to sums bhejne se pehle features ko **clip** karo aur har sum mein **Gaussian noise** daalo. Phir koi bhi ek sample ka asar mathematically chhupa rehta hai."]));
C.push(H2("Kaise kaam karta hai"));
C.push(B("**Clip:** har feature ka size ‖h‖ ≤ C (C = public held-out data ke norms ka 90th percentile). Isse ek sample ka max asar bounded ho jaata hai."));
C.push(B("**Noise:** chaaron releases (count, Σh, Σ‖h‖², Σφ) mein Gaussian noise; total privacy budget zCDP mein chaaron mein barabar baanta."));
C.push(B("**Gate:** jis class ka noise signal se zyada hai (bahut rare class), uske liye MC/KH use nahi karte, generator ka apna output rakhte hain."));
C.push(eq("eq_dp"));
C.push(H2("Result (long tail, 3 seeds)"));
C.push(...table(["Dataset", "GeFL-F (koi formal guarantee nahi)", "Ours-A exact", "Ours-A, ε = 8", "Ours-A, ε = 2"], [
  ["MNIST", "75.25", "94.82", "90.36", "85.06"],
  ["FashionMNIST", "58.47", "79.67", "72.77", "65.59"],
], [2.0, 2.6, 1.6, 1.6, 1.6]));
C.push(P("ε = 8 pe 4.5–6.9 points ka cost, zyada tar rarest classes pe (unke paas kam samples, to noise ka relative asar zyada). Phir bhi GeFL-F se +7 se +15 upar, jiske paas koi formal guarantee hi nahi."));
C.push(H2("Architecture mein kahan"));
C.push(...card("Detail", "Value", [
  ["**Kahan**", "Section 1 ke upload mein, client side (clip + noise), sirf ek baar"],
  ["**Baaki pipeline**", "Bilkul same; PC-VAE, MC, KH noisy statistics use karte hain"],
  ["**Code**", "_STAT_DP aur anchor_stats (experiments/core.py:1117, 1132), zcdp_rho (core.py:1127)"],
]));
yaad(["1) Clip + Gaussian noise, ek hi baar.", "2) ε = 8: MNIST 90.4, FMNIST 72.8.", "3) Optional hai; default secure aggregation."]);
