import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def add_trivial_features(df: pd.DataFrame) -> None:
    df['slips_perc_pro'] = df.pro_slips / df.total_slips
    df['bipartisan'] = (df.dem_sponsors > 0) & (df.rep_sponsors > 0)
    df['bipartisan'] = df['bipartisan'].map(lambda x: 1 if x else 0)
    df['ideol_range'] = df.most_cons_sponsor_ideology - df.most_liberal_sponsor_ideology
    df['first_word_approp'] = df.description.map(lambda x: 1 if x.split(' ')[0] == 'Appropriates' else 0)
    df['author_is_chair'] = df['author_is_chair'].map(lambda x: 1 if x else 0)
    return df.fillna(0)

def handle_nulls(df: pd.DataFrame) -> None:
    return df.fillna(0)

def get_pg_engine(ip_address: str):
  return create_engine(f'postgresql://ubuntu@{ip_address}/ubuntu', echo=False)

def fetch_raw(ip_address: str) -> pd.DataFrame:
  aws_engine = get_pg_engine(ip_address)
  query_data = """
  with main as (
    select
      b.bill_id,
      -- author fields
      pa.people_id as author_id,
      ia.np_score as author_ideology,
      pa.party_id as author_party,
      array_agg(pca.role) @> '{Chair}' as author_is_chair,
      pda.years_senate as author_years_sen,
      pda.total_funding as author_total_funding,
      -- sponsor fields
      count(distinct s.people_id) as total_sponsors,
      count(distinct s.people_id) filter (where ps.party_id = 1) as dem_sponsors,
      count(distinct s.people_id) filter (where ps.party_id = 2) as rep_sponsors,
      min(ids.np_score) as most_liberal_sponsor_ideology,
      max(ids.np_score) as most_cons_sponsor_ideology,
      count(distinct s.people_id) filter (where pcs.role = 'Chair') as sponsor_chairs,
      -- text
      b.description,
      -- target
      b.third_reading
    from sen_bill b
    left join author a on a.bill_id = b.bill_id
    left join people pa on pa.people_id = a.people_id
    left join ideology ia on a.people_id = ia.people_id
    left join people_committee pca on pca.people_id = a.people_id
    left join people_detail pda on pda.people_id = a.people_id
    left join sponsor_at_second s on s.bill_id = b.bill_id
    left join people ps on ps.people_id = s.people_id
    left join ideology ids on ids.people_id = s.people_id
    left join people_committee pcs on pcs.people_id = s.people_id
    left join people_detail pds on pds.people_id = s.people_id
    group by b.bill_id, b.third_reading, ia.np_score, pa.party_id, pa.people_id, pda.years_senate, pda.total_funding, b.description
  ), agg_sponsors as (
    select
      b.bill_id,
      sum(pd.total_funding) as agg_funding_sponsors,
      sum(pd.years_senate) as agg_exp_sponsors
    from sen_bill b
    join sponsor_at_second s on s.bill_id = b.bill_id
    join people_detail pd on pd.people_id = s.people_id
    group by b.bill_id
  ), slips as (
  SELECT
    b.bill_id,
    count(w.index) as total_slips,
    count(w.index) filter (where w.position = 'PROP') as pro_slips,
    count(w.index) filter (where w.position = 'OPP') as opp_slips,
    count(w.index) filter (where w.position = 'NOPOS') as no_pos_slips
    FROM sen_bill b
      -- works b/c this join is 1x1 and does not duplicate records (should make index distinct)
      left JOIN history hs ON hs.bill_id = b.bill_id AND hs.action ~* '^second reading'::text AND hs.chamber = 'Senate'::text
      left JOIN witness_slip w on w.bill_number = b.bill_number and w.date <= to_date(hs.date, 'YYYY-MM-DD')
      group by b.bill_id
  ) select
      m.*,
      a.agg_funding_sponsors,
      a.agg_exp_sponsors,
      s.total_slips,
      s.pro_slips,
      s.opp_slips,
      no_pos_slips
  from main m
  left join agg_sponsors a on m.bill_id = a.bill_id
  left join slips s on s.bill_id = m.bill_id
  where m.author_id not in (1082, 1055, 1052);
  """

  return pd.read_sql(query_data, con=aws_engine)

def fetch_model_data(ip_address: str) -> pd.DataFrame:
  raw = fetch_raw(ip_address)
  return add_trivial_features(handle_nulls(raw))
