-- Migration 006: Agent Templates & Customer Agents
-- Run: psql -U smarttalker -d smarttalker -f 006_agent_templates.sql

CREATE TABLE IF NOT EXISTS agent_templates (
    id                  VARCHAR(64) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    slug                VARCHAR(100) UNIQUE NOT NULL,
    name_ar             VARCHAR(200) NOT NULL,
    name_en             VARCHAR(200) NOT NULL,
    description_ar      TEXT NOT NULL,
    description_en      TEXT NOT NULL,
    job_title_ar        VARCHAR(200),
    job_title_en        VARCHAR(200),
    category            VARCHAR(100) NOT NULL,
    icon_emoji          VARCHAR(10)  DEFAULT '🤖',
    color_accent        VARCHAR(7)   DEFAULT '#00D4AA',
    default_language    VARCHAR(10)  DEFAULT 'ar',
    default_personality VARCHAR(50)  DEFAULT 'professional',
    system_prompt       TEXT,
    kb_template         JSONB        DEFAULT '{}',
    is_published        BOOLEAN      DEFAULT FALSE,
    sort_order          INTEGER      DEFAULT 0,
    created_at          TIMESTAMP    DEFAULT NOW(),
    updated_at          TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS customer_agents (
    id                  VARCHAR(64) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    customer_id         VARCHAR(64) NOT NULL REFERENCES customers(id),
    template_id         VARCHAR(64) REFERENCES agent_templates(id),
    name_ar             VARCHAR(200) NOT NULL,
    name_en             VARCHAR(200),
    description         TEXT,
    photo_url           TEXT,
    photo_r2_key        TEXT,
    photo_preprocessed  BOOLEAN  DEFAULT FALSE,
    voice_id            TEXT,
    voice_cloned        BOOLEAN  DEFAULT FALSE,
    personality         VARCHAR(50)  DEFAULT 'professional',
    language            VARCHAR(10)  DEFAULT 'ar',
    kb_document_count   INTEGER  DEFAULT 0,
    kb_faq_count        INTEGER  DEFAULT 0,
    kb_status           VARCHAR(20)  DEFAULT 'empty',
    is_active           BOOLEAN  DEFAULT TRUE,
    channels            JSONB DEFAULT '{"widget":true,"whatsapp":false,"telegram":false}',
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_customer_agents_customer ON customer_agents(customer_id);
CREATE INDEX IF NOT EXISTS idx_customer_agents_template ON customer_agents(template_id);
CREATE INDEX IF NOT EXISTS idx_agent_templates_published ON agent_templates(is_published) WHERE is_published = TRUE;

-- Seed: Electronic Services Officer
INSERT INTO agent_templates (
    slug, name_ar, name_en,
    description_ar, description_en,
    job_title_ar, job_title_en,
    category, icon_emoji, color_accent,
    default_language, default_personality,
    system_prompt, kb_template,
    is_published, sort_order
) VALUES (
    'electronic-services-officer',
    'موظف مكتب الخدمات الإلكترونية',
    'Electronic Services Officer',
    'موظف رقمي ذكي يساعد المراجعين في إنجاز معاملاتهم الحكومية والإلكترونية بكل سهولة، ويرشدهم خطوة بخطوة خلال الإجراءات المطلوبة',
    'Smart digital employee helping visitors complete government and electronic transactions easily, guiding them step by step',
    'موظف خدمات إلكترونية',
    'Electronic Services Officer',
    'government_services',
    '🏛️', '#00D4AA', 'ar', 'professional',
    'أنت موظف في مكتب خدمات إلكترونية. تساعد المراجعين في إنجاز معاملاتهم. تسأل عن نوع المعاملة، توضح المستندات المطلوبة، تحدد الرسوم، وتوجه المراجع خطوة بخطوة. لا تنجز المعاملة فعلياً لكن توجه وترشد. تتحدث باللغة التي يختارها المراجع.',
    '{"suggested_docs":["قائمة الخدمات المتاحة وأوقات العمل","المستندات المطلوبة لكل خدمة","الرسوم والتعريفات","الأسئلة الشائعة"],"sample_faqs":[{"q":"ما هي أوقات العمل؟","a":"نعمل من الأحد إلى الخميس من 8 صباحاً حتى 4 مساءً"},{"q":"كيف أجدد رخصة القيادة؟","a":"تحتاج: الهوية الوطنية، الرخصة القديمة، شهادة فحص النظر. الرسوم: 20 ريال عماني"},{"q":"كم تستغرق المعاملة؟","a":"معظم المعاملات تُنجز في نفس اليوم"},{"q":"هل يمكنني إرسال المستندات عبر واتساب؟","a":"نعم، أرسل صورة واضحة لكل مستند"},{"q":"ما طرق الدفع المتاحة؟","a":"نقبل الدفع النقدي وبطاقات الائتمان والدفع الإلكتروني"}],"training_prompts":["كيف أجدد الإقامة؟","ما المستندات المطلوبة لتسجيل مركبة؟","كيف أحصل على شهادة لا ممانعة؟"]}',
    TRUE, 1
), (
    'legal-advisor',
    'المستشار القانوني',
    'Legal Advisor',
    'مستشار قانوني رقمي يقدم إرشادات قانونية أولية ويساعد في فهم الحقوق والإجراءات، مع التوضيح بأن المشورة النهائية تتطلب محامياً مختصاً',
    'Digital legal advisor providing initial guidance on rights and procedures, while noting final advice requires a specialized lawyer',
    'مستشار قانوني',
    'Legal Advisor',
    'legal',
    '⚖️', '#5B4FE8', 'ar', 'formal',
    'أنت مستشار قانوني رقمي. تقدم إرشادات قانونية أولية. توضح الحقوق والإجراءات المتاحة. دائماً تنوه أن المشورة النهائية تتطلب محامياً مرخصاً. لا تقدم تشخيصاً قانونياً نهائياً. تكون واضحاً وتتجنب المصطلحات المعقدة.',
    '{"suggested_docs":["مجالات الاستشارة القانونية","الأسئلة القانونية الشائعة","حقوق المستهلك","قانون العمل الأساسي","إجراءات التقاضي"],"sample_faqs":[{"q":"ما حقوقي إذا فُصلت من العمل؟","a":"يحق لك مكافأة نهاية الخدمة وإشعار مسبق وشهادة خبرة. إذا كان الفصل تعسفياً يحق لك تعويض إضافي"},{"q":"كيف أرفع شكوى ضد شركة؟","a":"قدم شكوى لوزارة التجارة أو هيئة حماية المستهلك. جمّع كل الوثائق والعقود أولاً"},{"q":"ما الفرق بين العقد الابتدائي وعقد البيع؟","a":"الابتدائي يحجز ويحدد الشروط، والنهائي ينقل الملكية رسمياً ويُسجل لدى الجهات المختصة"},{"q":"هل استشارتكم سرية؟","a":"نعم، جميع المحادثات سرية تماماً"},{"q":"هل تساعدون في قضايا المحاكم؟","a":"نقدم إرشادات أولية فقط. قضايا المحاكم تتطلب توكيل محامٍ مرخص"}],"training_prompts":["ما حقوقي القانونية كمستأجر؟","كيف أؤسس شركة ذات مسؤولية محدودة؟","ما إجراءات الحصول على حكم نفقة؟"]}',
    TRUE, 2
) ON CONFLICT (slug) DO NOTHING;
