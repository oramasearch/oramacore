use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};

pub fn date_to_timestamp(date: &str) -> Result<i64> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(date) {
        return Ok(dt.timestamp());
    }

    let supported_datetime_formats = vec![
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ];

    for format in supported_datetime_formats {
        if let Ok(naive_dt) = NaiveDateTime::parse_from_str(date, format) {
            let dt = Utc.from_utc_datetime(&naive_dt);
            return Ok(dt.timestamp());
        }
    }

    let supported_date_formats = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"];

    for format in supported_date_formats {
        if let Ok(naive_date) = NaiveDate::parse_from_str(date, format) {
            let naive_dt = naive_date
                .and_hms_opt(0, 0, 0)
                .with_context(|| "Failed to convert date to datetime")?;
            let dt = Utc.from_utc_datetime(&naive_dt);
            return Ok(dt.timestamp());
        }
    }

    anyhow::bail!("Unable to parse unsupported date format: {}", date)
}
