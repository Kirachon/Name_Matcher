use crate::models::{NormalizedPerson, Person};

pub fn normalize_text(input: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    // Remove diacritics by decomposing to NFD and filtering combining marks
    input
        .nfd()
        .filter(|c| {
            // unicode_normalization exposes a method on char extension traits via tables.
            // There's no is_mark on char; emulate by excluding combining marks range.
            !unicode_normalization::char::is_combining_mark(*c)
        })
        .collect::<String>()
        .to_lowercase()
        .trim()
        .to_string()
}

pub fn normalize_person(p: &Person) -> NormalizedPerson {
    NormalizedPerson {
        id: p.id,
        uuid: p.uuid.clone(),
        first_name: normalize_text(&p.first_name),
        middle_name: p.middle_name.as_ref().map(|m| normalize_text(m)),
        last_name: normalize_text(&p.last_name),
        birthdate: p.birthdate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Person;
    use chrono::NaiveDate;

    #[test]
    fn test_normalize_text_diacritics() {
        assert_eq!(normalize_text("Álvaro"), "alvaro");
        assert_eq!(normalize_text("ÉÉ"), "ee");
        assert_eq!(normalize_text("  José  "), "jose");
    }

    #[test]
    fn test_normalize_person() {
        let p = Person { id: 1, uuid: "u".into(), first_name: "Éva".into(), middle_name: None, last_name: "Łukasz".into(), birthdate: NaiveDate::from_ymd_opt(2000,1,2).unwrap() };
        let n = normalize_person(&p);
        assert_eq!(n.first_name, "eva");
        assert_eq!(n.last_name, "łukasz");
    }
}

